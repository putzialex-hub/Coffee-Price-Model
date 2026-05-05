"""Walk-forward backtesting with full quantile fit and WIS/DM/coverage metrics.

Phase 1 upgrades:
  • Monthly rolling windows from 2018 (vs. quarterly from 2022 previously)
  • Naive random-walk baseline with historical-vol interval at every cutoff
  • ICE front-month futures price as market-implied benchmark (via yfinance)
  • Weighted Interval Score (WIS) — proper probabilistic accuracy metric
  • Diebold-Mariano test against naive baseline (statistical significance)
  • compare_baselines() prints a formatted comparison table
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import COMMODITIES, HORIZONS, QUANTILES
from .features import all_feature_columns, get_commodity_features, fill_remaining_nan_with_zero
from .models import _build_estimator

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


# ── scorers ──────────────────────────────────────────────────────────────────

def _pinball_loss(actual: float, pred: float, alpha: float) -> float:
    diff = actual - pred
    return alpha * diff if diff >= 0 else (alpha - 1) * diff


def wis_score(actual: float, q05: float, q50: float, q95: float) -> float:
    """Weighted Interval Score for a 90% prediction interval.

    WIS = 0.5*|y-median| + 0.05*(q95-q05) + (q05-y)₊ + (y-q95)₊

    Equivalent to CRPS for piecewise-linear CDFs. Lower is better. Units = price.
    The 0.05 coefficient is α/2 where α=0.10 (nominal non-coverage probability).
    The violation penalties (q05-y)₊ and (y-q95)₊ dominate when actual is outside
    the interval, making WIS a strict proper scoring rule.
    """
    mae_term = 0.5 * abs(actual - q50)
    sharpness = 0.05 * (q95 - q05)
    penalty_low = max(0.0, q05 - actual)
    penalty_high = max(0.0, actual - q95)
    return mae_term + sharpness + penalty_low + penalty_high


# ── ICE futures benchmark ─────────────────────────────────────────────────────

def _load_futures_benchmark(start: str = "2016-01-01") -> dict[str, pd.Series]:
    """Download ICE front-month futures as market-implied price benchmark.

    KC=F  = Coffee C Arabica (ICE US, USD cents/lb)  → arabica_price
    RC=F  = Robusta Coffee   (ICE EU, USD/tonne)     → robusta_price

    Front-month prices are a reasonable proxy for what markets expected prices
    to be at each point in time. Returns {} silently on any failure.
    """
    if not _YF_AVAILABLE:
        return {}
    try:
        import logging
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        ticker_map = {"KC=F": "arabica_price", "RC=F": "robusta_price"}
        out: dict[str, pd.Series] = {}
        for ticker, comm in ticker_map.items():
            raw = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw = raw.droplevel(1, axis=1)
            s = raw["Close"].copy()
            s.index = pd.to_datetime(s.index)
            s.name = comm
            out[comm] = s
        return out
    except Exception:
        return {}


# ── walk-forward backtesting ──────────────────────────────────────────────────

def walk_forward(
    df: pd.DataFrame,
    engine: str,
    val_start: str = "2018-01-01",
    val_step: str = "ME",
    val_dates: pd.DatetimeIndex | None = None,
    commodities: list[str] = COMMODITIES,
    horizons: list[int] = HORIZONS,
    quantiles: list[float] = QUANTILES,
    min_history_days: int = 365,
    min_train_rows: int = 500,
    include_futures_baseline: bool = True,
) -> pd.DataFrame:
    """Monthly rolling walk-forward from val_start through df.date.max()-max_horizon.

    For each cutoff date and each (commodity, horizon) we fit all three quantile
    models — identical configuration to the live model — and record:
      • model quantile predictions + WIS score
      • naive random-walk baseline with historical-volatility interval + WIS
      • ICE front-month futures price (when yfinance is reachable)

    Returns long-format DataFrame: one row per (cutoff, commodity, horizon).

    Parameters
    ----------
    val_start : first cutoff date (default 2018-01-01 for 7-year evaluation window)
    val_step  : pandas frequency string for cutoff spacing (default 'ME' = month-end)
    val_dates : explicit cutoff array — overrides val_start / val_step (backward compat)
    """
    if val_dates is None:
        max_cutoff = df["date"].max() - pd.Timedelta(days=max(horizons) + 5)
        val_dates = pd.date_range(val_start, max_cutoff, freq=val_step)

    # Load ICE futures benchmark (graceful no-op when network is unavailable)
    futures_data: dict[str, pd.Series] = {}
    if include_futures_baseline:
        futures_data = _load_futures_benchmark(start=val_start)
        if futures_data:
            loaded = [c.replace("_price", "") for c in futures_data]
            print(f"   📈 ICE Futures-Benchmark geladen: {', '.join(loaded)}")

    rows = []
    n_total = len(val_dates)
    for i, val_cutoff in enumerate(val_dates):
        val_cutoff = pd.Timestamp(val_cutoff)
        if val_cutoff < df["date"].min() + pd.Timedelta(days=min_history_days):
            continue
        if val_cutoff + pd.Timedelta(days=max(horizons)) > df["date"].max():
            continue

        train_data = df[df["date"] <= val_cutoff].copy()
        if len(train_data) < min_train_rows:
            continue

        # Progress indicator every 10 cutoffs
        if i % 10 == 0:
            print(f"   [{i+1}/{n_total}] Cutoff {val_cutoff.date()} …", end="\r", flush=True)

        for h in horizons:
            for comm in commodities:
                train_data[f"target_{comm}_{h}d"] = np.log(
                    train_data[comm].shift(-h) / train_data[comm]
                )

        if engine == "sklearn":
            train_data = fill_remaining_nan_with_zero(train_data)

        feature_cols_val = all_feature_columns(train_data)

        for comm in commodities:
            fcols = get_commodity_features(feature_cols_val, comm)
            for h in horizons:
                target = f"target_{comm}_{h}d"
                clean = train_data.dropna(subset=[target, comm])
                if len(clean) < 100:
                    continue
                X_val = clean[fcols]
                y_val = clean[target]
                last_row_val = clean.iloc[[-1]][fcols]
                start_price = float(clean.iloc[-1][comm])

                target_date = val_cutoff + pd.Timedelta(days=h)
                actual_df = df[df["date"] >= target_date]
                if len(actual_df) == 0:
                    continue
                actual_price = float(actual_df.iloc[0][comm])

                # ── model quantile predictions ────────────────────────────
                q_preds_log: dict[float, float] = {}
                for q in quantiles:
                    est = _build_estimator(comm, q, engine)
                    est.fit(X_val, y_val)
                    q_preds_log[q] = float(est.predict(last_row_val)[0])

                sorted_qs = sorted(quantiles)
                sorted_logs = sorted(q_preds_log[q] for q in sorted_qs)
                price_by_q = {q: start_price * np.exp(lp)
                              for q, lp in zip(sorted_qs, sorted_logs)}

                base_q = 0.5 if 0.5 in quantiles else sorted_qs[len(sorted_qs) // 2]
                pred_price = price_by_q[base_q]
                q05_price = price_by_q[min(quantiles)]
                q95_price = price_by_q[max(quantiles)]

                error_pct = (pred_price - actual_price) / actual_price * 100
                pinball_base = _pinball_loss(actual_price, pred_price, base_q)
                pinball_low = _pinball_loss(actual_price, q05_price, min(quantiles))
                pinball_high = _pinball_loss(actual_price, q95_price, max(quantiles))
                in_interval = int(q05_price <= actual_price <= q95_price)
                dir_correct = int((pred_price > start_price) == (actual_price > start_price))
                model_wis = wis_score(actual_price, q05_price, pred_price, q95_price)

                # ── naive random-walk baseline ────────────────────────────
                # Interval derived from historical daily log-return volatility,
                # scaled to horizon h (Gaussian RW approximation).
                log_rets = np.log(clean[comm] / clean[comm].shift(1)).dropna()
                sigma_h = log_rets.std() * np.sqrt(h)
                naive_pred = start_price
                naive_q05 = start_price * np.exp(-1.645 * sigma_h)
                naive_q95 = start_price * np.exp(+1.645 * sigma_h)
                naive_error_pct = (naive_pred - actual_price) / actual_price * 100
                naive_wis = wis_score(actual_price, naive_q05, naive_pred, naive_q95)

                # ── ICE futures front-month baseline ──────────────────────
                futures_pred = np.nan
                futures_error_pct = np.nan
                if comm in futures_data:
                    fut_s = futures_data[comm]
                    avail = fut_s[fut_s.index <= val_cutoff]
                    if not avail.empty:
                        futures_pred = float(avail.iloc[-1])
                        futures_error_pct = (
                            (futures_pred - actual_price) / actual_price * 100
                        )

                rows.append({
                    "cutoff": val_cutoff,
                    "commodity": comm,
                    "horizon": h,
                    "start_price": start_price,
                    "predicted": pred_price,
                    "predicted_low": q05_price,
                    "predicted_high": q95_price,
                    "actual": actual_price,
                    "error_pct": error_pct,
                    "pinball_base": pinball_base,
                    "pinball_low": pinball_low,
                    "pinball_high": pinball_high,
                    "in_interval": in_interval,
                    "dir_correct": dir_correct,
                    "model_wis": model_wis,
                    "naive_pred": naive_pred,
                    "naive_error_pct": naive_error_pct,
                    "naive_wis": naive_wis,
                    "futures_pred": futures_pred,
                    "futures_error_pct": futures_error_pct,
                })

    print()  # newline after progress line
    return pd.DataFrame(rows)


def summarise(val_df: pd.DataFrame) -> dict[tuple, dict]:
    """Return per (commodity, horizon) summary metrics including WIS skill score."""
    out: dict[tuple, dict] = {}
    if val_df.empty:
        return out
    for comm in val_df["commodity"].unique():
        for h in sorted(val_df["horizon"].unique()):
            sub = val_df[(val_df["commodity"] == comm) & (val_df["horizon"] == h)]
            if sub.empty:
                continue
            dir_acc = sub["dir_correct"].mean() * 100 if "dir_correct" in sub else float("nan")
            model_wis_mean = sub["model_wis"].mean() if "model_wis" in sub.columns else float("nan")
            naive_wis_mean = sub["naive_wis"].mean() if "naive_wis" in sub.columns else float("nan")
            if naive_wis_mean > 0 and not np.isnan(naive_wis_mean) and not np.isnan(model_wis_mean):
                wis_skill = (1.0 - model_wis_mean / naive_wis_mean) * 100
            else:
                wis_skill = float("nan")
            out[(comm, h)] = {
                "mae": sub["error_pct"].abs().mean(),
                "bias": sub["error_pct"].mean(),
                "std": sub["error_pct"].std(),
                "hit_rate_15": (sub["error_pct"].abs() < 15).mean() * 100,
                "dir_accuracy": dir_acc,
                "coverage_90": sub["in_interval"].mean() * 100,
                "pinball_base": sub["pinball_base"].mean(),
                "pinball_low": sub["pinball_low"].mean(),
                "pinball_high": sub["pinball_high"].mean(),
                "model_wis": model_wis_mean,
                "naive_wis": naive_wis_mean,
                "wis_skill": wis_skill,
                "n": len(sub),
            }
    return out


def bias_corrections(val_df: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Mean log-return error per (commodity, horizon), used as a live offset.

    We compute the log-space bias (predicted/actual) and apply it as a
    multiplicative correction. The 6.4% Robusta bias documented in v5 motivates
    this; the correction is small and is logged in the output CSV.
    """
    out: dict[tuple[str, int], float] = {}
    if val_df.empty:
        return out
    for comm in val_df["commodity"].unique():
        for h in sorted(val_df["horizon"].unique()):
            sub = val_df[(val_df["commodity"] == comm) & (val_df["horizon"] == h)]
            if sub.empty:
                continue
            log_bias = np.mean(np.log(sub["predicted"] / sub["actual"]))
            out[(comm, int(h))] = float(log_bias)
    return out


def conformal_deltas(
    val_df: pd.DataFrame,
    target_coverage: float = 0.90,
) -> dict[tuple[str, int], float]:
    """Compute per-(commodity, horizon) conformal expansion delta.

    Uses split-conformal prediction for quantile regression:
    - Nonconformity score s_i = max(q_low_i - actual_i, actual_i - q_high_i)
      (positive when actual is outside the raw interval, negative when inside)
    - q_hat = the ceil((n+1)*(1-alpha))/n empirical quantile of scores
    - At prediction time: adjusted_low = q_low - q_hat,
                          adjusted_high = q_high + q_hat

    This guarantees ~target_coverage marginal coverage (finite-sample valid).
    When the raw interval is already wide enough, q_hat may be negative,
    which slightly tightens the interval — that is intentional.
    """
    out: dict[tuple[str, int], float] = {}
    if val_df.empty:
        return out
    alpha = 1.0 - target_coverage
    for comm in val_df["commodity"].unique():
        for h in sorted(val_df["horizon"].unique()):
            sub = val_df[(val_df["commodity"] == comm) & (val_df["horizon"] == h)].copy()
            if len(sub) < 4:
                out[(comm, int(h))] = 0.0
                continue
            scores = np.maximum(
                sub["predicted_low"].values - sub["actual"].values,
                sub["actual"].values - sub["predicted_high"].values,
            )
            n = len(scores)
            level = np.ceil((n + 1) * (1 - alpha)) / n
            level = min(level, 1.0)
            q_hat = float(np.quantile(scores, level))
            out[(comm, int(h))] = q_hat
    return out


def apply_conformal(
    preds: dict[tuple, float],
    deltas: dict[tuple[str, int], float],
    commodities: list[str],
    horizons: list[int],
) -> dict[tuple, float]:
    """Expand/contract Low and High predictions by conformal delta."""
    corrected = dict(preds)
    for comm in commodities:
        for h in horizons:
            delta = deltas.get((comm, int(h)), 0.0)
            corrected[(comm, h, 0.05)] = preds[(comm, h, 0.05)] - delta
            corrected[(comm, h, 0.95)] = preds[(comm, h, 0.95)] + delta
    return corrected


# ── Diebold-Mariano test ──────────────────────────────────────────────────────

def diebold_mariano_test(
    losses_model: np.ndarray,
    losses_baseline: np.ndarray,
    h: int = 1,
) -> tuple[float, float]:
    """Diebold-Mariano test for equal predictive accuracy (two-sided).

    d_t = loss_model_t - loss_baseline_t
    DM  = d̄ / sqrt(HAC_var(d) / n)   ~  N(0,1) asymptotically

    Uses Newey-West HAC variance with lag = h (rule of thumb for h-step forecasts).
    Returns (dm_stat, p_value). Negative DM_stat means model has lower loss (wins).
    """
    d = np.asarray(losses_model, dtype=float) - np.asarray(losses_baseline, dtype=float)
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    d_bar = d.mean()

    # Newey-West HAC variance estimator.
    # Rule of thumb: lag ≤ min(h, floor(n^(1/3))) keeps the estimator stable
    # when n is small (avoids over-correcting with too many autocovariance lags).
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    max_lag = min(h, max(1, int(np.floor(n ** (1 / 3)))))
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        gamma_l = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        if np.isnan(gamma_l):
            continue
        gamma_sum += 2.0 * weight * gamma_l

    hac_var = (gamma_0 + gamma_sum) / n
    if hac_var <= 0:
        return float("nan"), float("nan")

    dm_stat = d_bar / np.sqrt(hac_var)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


# ── baseline comparison table ─────────────────────────────────────────────────

def compare_baselines(val_df: pd.DataFrame) -> pd.DataFrame:
    """DM test per (commodity, horizon) — model vs naive random-walk.

    Also includes ICE futures baseline when available. Returns a DataFrame
    and prints a formatted table to stdout.
    """
    if val_df.empty:
        return pd.DataFrame()

    has_futures = (
        "futures_error_pct" in val_df.columns
        and val_df["futures_error_pct"].notna().any()
    )

    rows = []
    for comm in sorted(val_df["commodity"].unique()):
        for h in sorted(val_df["horizon"].unique()):
            sub = val_df[
                (val_df["commodity"] == comm) & (val_df["horizon"] == h)
            ].dropna(subset=["error_pct", "naive_error_pct"])
            if len(sub) < 10:
                continue

            losses_model = sub["error_pct"].abs().values
            losses_naive = sub["naive_error_pct"].abs().values
            dm_stat, p_value = diebold_mariano_test(losses_model, losses_naive, h=h)

            model_mae = losses_model.mean()
            naive_mae = losses_naive.mean()
            skill_pct = (1.0 - model_mae / naive_mae) * 100 if naive_mae > 0 else float("nan")

            model_wis = sub["model_wis"].mean() if "model_wis" in sub.columns else float("nan")
            naive_wis = sub["naive_wis"].mean() if "naive_wis" in sub.columns else float("nan")
            wis_skill = (
                (1.0 - model_wis / naive_wis) * 100
                if naive_wis > 0 and not np.isnan(naive_wis)
                else float("nan")
            )

            futures_mae = float("nan")
            if has_futures:
                fut_sub = sub.dropna(subset=["futures_error_pct"])
                if len(fut_sub) >= 5:
                    futures_mae = fut_sub["futures_error_pct"].abs().mean()

            rows.append({
                "commodity": comm.replace("_price", ""),
                "horizon": h,
                "model_mae": model_mae,
                "naive_mae": naive_mae,
                "futures_mae": futures_mae,
                "mae_skill_pct": skill_pct,
                "model_wis": model_wis,
                "naive_wis": naive_wis,
                "wis_skill_pct": wis_skill,
                "dm_stat": dm_stat,
                "p_value": p_value,
                "significant_10pct": not np.isnan(p_value) and p_value < 0.10,
                "n": len(sub),
            })

    result = pd.DataFrame(rows)

    if not result.empty:
        has_fut_col = result["futures_mae"].notna().any()
        print("\n" + "=" * 82)
        print("BASELINE COMPARISON — Model vs. Naive Random-Walk (DM Test)")
        print("=" * 82)
        hdr = (f"{'Comm':8} {'H':>4} | {'Model MAE':>9} {'Naive MAE':>9}"
               + (f" {'Futures':>9}" if has_fut_col else "")
               + f" | {'Skill':>6} {'WIS-Skill':>9} | {'DM stat':>7} {'p-val':>6} Sig | n")
        print(hdr)
        print("-" * 82)
        for _, r in result.iterrows():
            fut_str = (f" {r['futures_mae']:>9.1f}" if has_fut_col else "")
            sig = "★" if r["significant_10pct"] else " "
            dm_s = f"{r['dm_stat']:>7.2f}" if not np.isnan(r["dm_stat"]) else f"{'—':>7}"
            p_s = f"{r['p_value']:>6.3f}" if not np.isnan(r["p_value"]) else f"{'—':>6}"
            sk = f"{r['mae_skill_pct']:>+5.1f}%" if not np.isnan(r["mae_skill_pct"]) else f"{'—':>6}"
            wsk = f"{r['wis_skill_pct']:>+8.1f}%" if not np.isnan(r["wis_skill_pct"]) else f"{'—':>9}"
            print(f"{r['commodity']:8} {r['horizon']:>4}d | "
                  f"{r['model_mae']:>9.1f} {r['naive_mae']:>9.1f}{fut_str} | "
                  f"{sk} {wsk} | {dm_s} {p_s}  {sig}  | {r['n']}")
        print("-" * 82)
        print("Skill > 0 = model beats naive. DM: negative stat = lower loss. ★ p < 0.10")
    return result
