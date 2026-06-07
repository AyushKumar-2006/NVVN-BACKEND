"""Core synthetic-load assembly for Chhattisgarh (pure numpy/pandas, no Django).

Pipeline (see __init__.py for the philosophy):

    base(month,dow,slot)              # learned from real data — seasonal/weekly/
                                      #   intraday/industrial/agri structure
      x year_scale                    # YoY growth anchor (2022 = 2024-6%, 2023 +3%)
      x festival_multiplier           # CG festival calendar overlay
      + weather_anomaly_delta         # mean-centred AC/heating response to real temp
      x (1 + ar1_noise + daily_drift) # smooth ±2-4% wander (NOT white noise)
      + grid_events                   # rare spikes / partial outages
      -> enforce(no-neg, |Δ5min|<=300 MW)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .events import festival_multiplier, EVENING_HOURS

SLOTS_PER_DAY = 288


# --------------------------------------------------------------------------- #
@dataclass
class GenConfig:
    # year-on-year growth (relative to the real 2024 level)
    growth_2022: float = 0.06          # 2022 = 2024 level - 6%
    growth_2023: float = 0.03          # 2023 = 2022 + 3%
    ref_year: int = 2024
    # smooth AR(1) noise: stationary std ~3% with high persistence so 5-min
    # increments stay realistically small (real CG p99 jump ~55 MW)
    noise_std: float = 0.03
    noise_phi: float = 0.985
    # slow within-day drift envelope (±1%)
    drift_max_frac: float = 0.01
    # weather sensitivity (MW per deg C beyond comfort band), applied to anomaly
    ac_mw_per_deg: float = 50.0        # > 30 C
    heat_mw_per_deg: float = 30.0      # < 15 C
    comfort_hi: float = 30.0
    comfort_lo: float = 15.0
    weather_clip_mw: float = 400.0
    # rare events
    p_grid_event: float = 0.001        # 0.1% blocks: spike
    grid_event_mw: tuple = (150.0, 280.0)
    p_outage: float = 0.005            # 0.5% blocks: small rural dip
    outage_mw: tuple = (50.0, 250.0)
    # seasonal peak-demand events: hot summer afternoons ride above the smooth
    # climatological profile (this is where the real annual peak comes from), so
    # without them the synthetic upper tail is too compressed (load factor > 0.75)
    peak_months: tuple = (3, 4, 5, 6)
    p_peak_day: float = 0.10           # ~10% of summer days see a heat peak
    peak_amp: tuple = (0.06, 0.14)     # fractional afternoon bump
    peak_center_hour: float = 16.0
    peak_sigma_h: float = 3.0
    # hard validation constraints
    max_jump_mw: float = 300.0
    lf_lo: float = 0.65
    lf_hi: float = 0.75
    monthly_tol: float = 0.05          # +/-5% vs CEA-proxy (real climatology)
    # profile learning
    smooth_slots: int = 5              # ~25-min circular smoothing of intraday curve
    min_count: int = 3
    seed: int = 42


# --------------------------------------------------------------------------- #
# Profile learning
# --------------------------------------------------------------------------- #
def _slot(idx):
    return idx.hour * 12 + idx.minute // 5


def _circ_smooth(a, win):
    """Circular moving average along the last axis (wraps midnight)."""
    if win <= 1:
        return a
    k = np.ones(win) / win
    pad = win
    ext = np.concatenate([a[..., -pad:], a, a[..., :pad]], axis=-1)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), -1, ext)
    return out[..., pad:-pad]


def learn_profile(real_df: pd.DataFrame, cfg: GenConfig):
    """Build hierarchical (month,dow,slot) base tables + real climatology.

    Returns a dict with numpy lookup arrays and reference stats used downstream.
    """
    df = real_df.dropna(subset=["load_mw"]).copy()
    idx = pd.DatetimeIndex(df["datetime"])
    df["month"] = idx.month
    df["dow"] = idx.dayofweek
    df["slot"] = _slot(idx)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["year"] = idx.year
    y = df["load_mw"].astype(float)

    # level-0: (month, dow, slot)
    lvl0 = np.full((12, 7, SLOTS_PER_DAY), np.nan)
    cnt0 = np.zeros((12, 7, SLOTS_PER_DAY))
    g0 = y.groupby([df["month"], df["dow"], df["slot"]]).agg(["mean", "size"])
    for (m, d, s), row in g0.iterrows():
        lvl0[m - 1, d, s] = row["mean"]
        cnt0[m - 1, d, s] = row["size"]

    # level-1: (month, is_weekend, slot)
    lvl1 = np.full((12, 2, SLOTS_PER_DAY), np.nan)
    g1 = y.groupby([df["month"], df["is_weekend"], df["slot"]]).mean()
    for (m, w, s), v in g1.items():
        lvl1[m - 1, w, s] = v

    # level-2: (month, slot)  and level-3: (slot)
    lvl2 = np.full((12, SLOTS_PER_DAY), np.nan)
    for (m, s), v in y.groupby([df["month"], df["slot"]]).mean().items():
        lvl2[m - 1, s] = v
    lvl3 = np.full(SLOTS_PER_DAY, np.nan)
    for s, v in y.groupby(df["slot"]).mean().items():
        lvl3[s] = v

    # smooth the intraday curve at the noisiest levels
    lvl0 = _circ_smooth(lvl0, cfg.smooth_slots)
    lvl2 = _circ_smooth(lvl2, cfg.smooth_slots)

    monthly_clim = y.groupby(df["month"]).mean()          # CEA proxy
    annual_mean = y.groupby(df["year"]).mean()

    return {
        "lvl0": lvl0, "cnt0": cnt0, "lvl1": lvl1, "lvl2": lvl2, "lvl3": lvl3,
        "monthly_clim": monthly_clim, "annual_mean": annual_mean,
        "global_mean": float(y.mean()), "global_min": float(y.min()),
        "global_max": float(y.max()),
    }


def base_series(index: pd.DatetimeIndex, profile: dict, cfg: GenConfig) -> np.ndarray:
    """Vectorised base load for each slot, with hierarchical fallback."""
    m = index.month.values - 1
    d = index.dayofweek.values
    w = (index.dayofweek.values >= 5).astype(int)
    s = _slot(index)

    v = profile["lvl0"][m, d, s]
    c = profile["cnt0"][m, d, s]
    use1 = np.isnan(v) | (c < cfg.min_count)
    v = np.where(use1, profile["lvl1"][m, w, s], v)
    use2 = np.isnan(v)
    v = np.where(use2, profile["lvl2"][m, s], v)
    use3 = np.isnan(v)
    v = np.where(use3, profile["lvl3"][s], v)
    return v.astype("float64")


# --------------------------------------------------------------------------- #
# Overlays
# --------------------------------------------------------------------------- #
def year_scales(index: pd.DatetimeIndex, base: np.ndarray, profile: dict, cfg: GenConfig):
    """Per-row multiplicative scale so each year's mean hits its growth target."""
    ref_mean = float(profile["annual_mean"].get(cfg.ref_year, profile["global_mean"]))
    targets = {
        2022: ref_mean * (1 - cfg.growth_2022),
        2023: ref_mean * (1 - cfg.growth_2022) * (1 + cfg.growth_2023),
    }
    years = index.year.values
    scale = np.ones(len(index))
    info = {}
    for yr, tgt in targets.items():
        mask = years == yr
        if not mask.any():
            continue
        pred = base[mask].mean()
        sc = tgt / pred if pred else 1.0
        scale[mask] = sc
        info[yr] = {"target_mean": tgt, "base_pred_mean": float(pred), "scale": float(sc)}
    return scale, info


def weather_anomaly_delta(index: pd.DatetimeIndex, temp_hourly: pd.Series, cfg: GenConfig):
    """Mean-centred MW delta from real temperature (preserves monthly means).

    raw = AC/heating response to absolute temp; we subtract the per-(month,hour)
    mean so only the *anomaly* (this hour hotter/cooler than typical) moves load.
    """
    if temp_hourly is None or len(temp_hourly) == 0:
        return np.zeros(len(index)), False
    th = temp_hourly.sort_index()
    # nearest-hour temperature for every 5-min slot
    hourly_key = index.floor("h")
    temp = th.reindex(hourly_key, method="nearest", tolerance=pd.Timedelta("3h")).values
    temp = pd.Series(temp).ffill().bfill().values
    raw = (cfg.ac_mw_per_deg * np.clip(temp - cfg.comfort_hi, 0, None)
           + cfg.heat_mw_per_deg * np.clip(cfg.comfort_lo - temp, 0, None))
    # centre per (month, hour)
    key = pd.DataFrame({"raw": raw, "m": index.month.values, "h": index.hour.values})
    mean_mh = key.groupby(["m", "h"])["raw"].transform("mean")
    delta = raw - mean_mh.values
    return np.clip(delta, -cfg.weather_clip_mw, cfg.weather_clip_mw), True


def ar1_noise(n: int, cfg: GenConfig, rng: np.random.Generator) -> np.ndarray:
    """Smooth multiplicative noise factor (1 + x), x ~ AR(1) with std=noise_std."""
    from scipy.signal import lfilter
    inn = cfg.noise_std * np.sqrt(1 - cfg.noise_phi ** 2)
    x = lfilter([inn], [1.0, -cfg.noise_phi], rng.standard_normal(n))
    return 1.0 + x


def daily_drift(index: pd.DatetimeIndex, cfg: GenConfig, rng: np.random.Generator) -> np.ndarray:
    """Slow within-day sinusoidal drift, ±drift_max_frac, random amp/phase per day."""
    day_id, _ = pd.factorize(index.normalize())
    n_days = day_id.max() + 1
    amp = rng.uniform(0.3 * cfg.drift_max_frac, cfg.drift_max_frac, n_days)
    phase = rng.uniform(0, 2 * np.pi, n_days)
    frac = (index.hour.values * 3600 + index.minute.values * 60) / 86400.0
    return amp[day_id] * np.sin(2 * np.pi * frac + phase[day_id])


def grid_events(n: int, cfg: GenConfig, rng: np.random.Generator) -> np.ndarray:
    """Additive MW: rare grid spikes (+/-) and small rural partial outages (-)."""
    add = np.zeros(n)
    spike = rng.random(n) < cfg.p_grid_event
    mag = rng.uniform(*cfg.grid_event_mw, n) * rng.choice([-1.0, 1.0], n)
    add[spike] += mag[spike]
    outage = rng.random(n) < cfg.p_outage
    add[outage] -= rng.uniform(*cfg.outage_mw, n)[outage]
    return add


def seasonal_peak_events(index, base, cfg: GenConfig, rng):
    """Additive MW: smooth afternoon heat-peak bumps on a few summer days.

    Lifts the annual peak realistically (the real grid peak occurs on the hottest
    summer afternoons, above the average profile) without disturbing monthly means.
    """
    add = np.zeros(len(index))
    day_id, day_first = pd.factorize(index.normalize())
    day_month = pd.DatetimeIndex(day_first).month.values
    is_summer = np.isin(day_month, cfg.peak_months)
    pick = is_summer & (rng.random(len(day_first)) < cfg.p_peak_day)
    if not pick.any():
        return add
    amp = np.where(pick, rng.uniform(*cfg.peak_amp, len(day_first)), 0.0)
    hour_frac = index.hour.values + index.minute.values / 60.0
    shape = np.exp(-0.5 * ((hour_frac - cfg.peak_center_hour) / cfg.peak_sigma_h) ** 2)
    return base * amp[day_id] * shape


def enforce_constraints(arr: np.ndarray, floor: float, max_jump: float) -> np.ndarray:
    """Floor at `floor` (no negatives) then clamp |Δ| between blocks to max_jump."""
    out = np.maximum(arr, floor).astype("float64")
    # forward pass: limit step-to-step change
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        if delta > max_jump:
            out[i] = out[i - 1] + max_jump
        elif delta < -max_jump:
            out[i] = out[i - 1] - max_jump
    return np.maximum(out, floor)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate(real_df, start, end, cfg: GenConfig, temp_hourly=None,
             seam_target=None, seam_slots=24):
    """Return (synth_df[datetime,load_mw,temperature_c], meta dict).

    If `seam_target` (the real load at the first slot after `end`) is given, the
    final `seam_slots` are tapered toward it so the synthetic series joins the
    real series without a discontinuity (keeps |Δ5min| small across the seam).
    """
    rng = np.random.default_rng(cfg.seed)
    profile = learn_profile(real_df, cfg)

    index = pd.date_range(start=pd.Timestamp(start),
                          end=pd.Timestamp(end) + pd.Timedelta("23:55:00"),
                          freq="5min")

    base = base_series(index, profile, cfg)
    scale, growth_info = year_scales(index, base, profile, cfg)
    fest = festival_multiplier(index)
    wdelta, weather_used = weather_anomaly_delta(index, temp_hourly, cfg)
    noise = ar1_noise(len(index), cfg, rng)
    drift = daily_drift(index, cfg, rng)
    events = grid_events(len(index), cfg, rng)
    peak_bump = seasonal_peak_events(index, base * scale, cfg, rng)

    load = base * scale * fest
    load = load + wdelta + peak_bump
    load = load * (noise + drift)
    load = load + events

    # seam continuity: blend the tail toward the real first value so the
    # synthetic->real join has no >300 MW step
    if seam_target is not None and seam_slots > 0:
        n = len(load)
        k = min(seam_slots, n - 1)
        wgt = np.zeros(n)
        wgt[n - k:] = np.linspace(1.0 / k, 1.0, k)
        load = load * (1 - wgt) + float(seam_target) * wgt

    floor = max(500.0, 0.7 * profile["global_min"] * (1 - cfg.growth_2022))
    load = enforce_constraints(load, floor, cfg.max_jump_mw)
    load = np.round(load, 1)

    temp_col = None
    if weather_used:
        hk = index.floor("h")
        temp_col = (temp_hourly.sort_index()
                    .reindex(hk, method="nearest", tolerance=pd.Timedelta("3h"))
                    .ffill().bfill().round(2).values)

    synth = pd.DataFrame({"datetime": index, "load_mw": load})
    if temp_col is not None:
        synth["temperature_c"] = temp_col

    meta = {
        "profile": profile, "growth": growth_info, "weather_used": weather_used,
        "n_rows": len(synth), "floor": floor,
        "festival_dates": int(np.sum(fest != 1.0)),
    }
    return synth, meta


# --------------------------------------------------------------------------- #
# Validation (item 10) + comparison
# --------------------------------------------------------------------------- #
def expected_monthly(profile: dict, growth_info: dict, cfg: GenConfig):
    """CEA-proxy expected monthly mean = real climatology x that year's scale."""
    clim = profile["monthly_clim"]
    out = {}
    for yr, gi in growth_info.items():
        for m in range(1, 13):
            out[(yr, m)] = float(clim.get(m, profile["global_mean"]) * gi["scale"])
    return out


def validate(synth: pd.DataFrame, profile: dict, growth_info: dict, cfg: GenConfig):
    s = synth["load_mw"].astype(float)
    idx = pd.DatetimeIndex(synth["datetime"])
    checks = {}

    checks["no_negative"] = {"pass": bool(s.min() > 0), "min": float(s.min())}

    jumps = s.diff().abs()
    mx = float(jumps.max())
    checks["max_jump_300"] = {"pass": bool(mx <= cfg.max_jump_mw + 1e-6),
                              "max_jump_mw": mx}

    lf = float(s.mean() / s.max())
    checks["load_factor"] = {"pass": bool(cfg.lf_lo <= lf <= cfg.lf_hi),
                             "load_factor": lf, "band": [cfg.lf_lo, cfg.lf_hi]}

    exp = expected_monthly(profile, growth_info, cfg)
    ym = s.groupby([idx.year, idx.month]).mean()
    worst = 0.0
    worst_key = None
    rows = []
    for (yr, m), val in ym.items():
        e = exp.get((yr, m))
        if not e:
            continue
        dev = abs(val - e) / e
        rows.append((yr, m, float(val), e, dev))
        if dev > worst:
            worst, worst_key = dev, (yr, m)
    checks["monthly_within_5pct"] = {
        "pass": bool(worst <= cfg.monthly_tol),
        "worst_dev_pct": round(worst * 100, 2),
        "worst_month": worst_key, "rows": rows,
    }

    checks["all_pass"] = all(v["pass"] for k, v in checks.items() if k != "all_pass")
    return checks


def _describe(s: pd.Series):
    s = s.astype(float)
    return {
        "n": int(s.size), "mean": float(s.mean()), "std": float(s.std()),
        "min": float(s.min()), "p5": float(s.quantile(.05)),
        "p50": float(s.quantile(.50)), "p95": float(s.quantile(.95)),
        "max": float(s.max()), "load_factor": float(s.mean() / s.max()),
        "cov_pct": float(s.std() / s.mean() * 100),
    }


def compare(real_df: pd.DataFrame, synth_df: pd.DataFrame):
    """Return dicts of distribution stats + monthly/hourly/weekly means."""
    r = real_df.dropna(subset=["load_mw"]).copy()
    ri = pd.DatetimeIndex(r["datetime"]); rs = r["load_mw"].astype(float)
    s = synth_df.copy()
    si = pd.DatetimeIndex(s["datetime"]); ss = s["load_mw"].astype(float)

    out = {"overall": {"real": _describe(rs), "synth": _describe(ss)}}
    out["monthly"] = {
        "real": rs.groupby(ri.month).mean().round(0).to_dict(),
        "synth": ss.groupby(si.month).mean().round(0).to_dict(),
    }
    out["hourly"] = {
        "real": rs.groupby(ri.hour).mean().round(0).to_dict(),
        "synth": ss.groupby(si.hour).mean().round(0).to_dict(),
    }
    we_r = rs[ri.dayofweek >= 5].mean() / rs[ri.dayofweek < 5].mean() - 1
    we_s = ss[si.dayofweek >= 5].mean() / ss[si.dayofweek < 5].mean() - 1
    out["weekend_effect_pct"] = {"real": round(we_r * 100, 2),
                                 "synth": round(we_s * 100, 2)}
    return out
