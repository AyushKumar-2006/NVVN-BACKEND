"""CG-specific festival / observance calendar for synthetic-load shaping.

Each entry is a multiplicative effect applied on top of the learned base profile
for the given date(s). `scope="day"` multiplies the whole day; `scope="evening"`
multiplies only the evening block (hours in EVENING_HOURS). Effects are applied
on the exact 2022/2023 calendar dates (festivals drift year-to-year, so a smooth
month×dow average cannot place them — this overlay does).

NOTE on realism: the real CG load is industrial-baseload-heavy, so festival dips
are softened relative to a residential grid, but the requested CG-specific
effects are encoded here. Tweak factors/dates in one place.

Factor convention: 0.75 == -25%, 1.15 == +15%.
"""

from __future__ import annotations

from datetime import date, timedelta

EVENING_HOURS = range(18, 24)          # 18:00-23:55 inclusive
MORNING_HOURS = range(6, 11)           # used for "Monday ramp" style nudges


def _span(d1: date, d2: date):
    """Inclusive list of dates from d1..d2."""
    out, cur = [], d1
    while cur <= d2:
        out.append(cur)
        cur += timedelta(days=1)
    return out


# --------------------------------------------------------------------------- #
# Festival calendar — (date, factor, scope, label)
# Dates verified for 2022 & 2023 (Chhattisgarh / India).
# --------------------------------------------------------------------------- #
def _festival_events():
    ev = []

    def add(d, factor, scope, label):
        ev.append((d, factor, scope, label))

    # ---- fixed-date civic / CG observances (both years) ----
    for yr in (2022, 2023):
        add(date(yr, 1, 26), 0.90, "day", "Republic Day")          # -10%
        add(date(yr, 8, 15), 0.90, "day", "Independence Day")      # -10%
        add(date(yr, 11, 1), 0.88, "day", "CG Foundation Day")     # -12%
        add(date(yr, 12, 25), 0.85, "day", "Christmas")            # -15%
        add(date(yr, 12, 31), 0.85, "day", "New Year's Eve")       # -15%
        add(date(yr, 1, 1), 0.85, "day", "New Year's Day")         # -15%

    # ---- Holi (Dhulandi) -20% ----
    add(date(2022, 3, 18), 0.80, "day", "Holi")
    add(date(2023, 3, 8), 0.80, "day", "Holi")

    # ---- Eid-ul-Fitr & Eid-ul-Adha -10% ----
    for d in (date(2022, 5, 3), date(2022, 7, 10), date(2023, 4, 22), date(2023, 6, 29)):
        add(d, 0.90, "day", "Eid")

    # ---- Raksha Bandhan -10% ----
    add(date(2022, 8, 11), 0.90, "day", "Raksha Bandhan")
    add(date(2023, 8, 30), 0.90, "day", "Raksha Bandhan")

    # ---- Hareli / Gondi-tribal (CG) -8% (rural) ----
    add(date(2022, 7, 28), 0.92, "day", "Hareli (tribal)")
    add(date(2023, 8, 16), 0.92, "day", "Hareli (tribal)")

    # ---- Navratri +8% evening (garba), 9-night span ----
    for d in _span(date(2022, 9, 26), date(2022, 10, 5)):
        add(d, 1.08, "evening", "Navratri")
    for d in _span(date(2023, 10, 15), date(2023, 10, 24)):
        add(d, 1.08, "evening", "Navratri")

    # ---- Diwali: +15% day-before, then -25% for 2 days ----
    for main in (date(2022, 10, 24), date(2023, 11, 12)):
        add(main - timedelta(days=1), 1.15, "day", "Pre-Diwali")
        add(main, 0.75, "day", "Diwali")
        add(main + timedelta(days=1), 0.75, "day", "Diwali +1")

    return ev


FESTIVAL_EVENTS = _festival_events()


def festival_multiplier(index, evening_hours=EVENING_HOURS):
    """Return a float multiplier array aligned to a 5-min DatetimeIndex.

    Day-scope effects multiply the whole day; evening-scope effects multiply only
    EVENING_HOURS. If several effects touch the same slot they compound.
    """
    import numpy as np

    mult = np.ones(len(index), dtype="float64")
    dser = index.normalize()                       # date at midnight per slot
    hours = index.hour
    by_date = {}
    for d, factor, scope, _label in FESTIVAL_EVENTS:
        by_date.setdefault(d, []).append((factor, scope))

    # group slots by calendar date for vectorised masking
    import pandas as pd
    dts = pd.Series(dser.date, index=range(len(index)))
    for d, effects in by_date.items():
        day_mask = (dts.values == d)
        if not day_mask.any():
            continue
        for factor, scope in effects:
            if scope == "day":
                mult[day_mask] *= factor
            elif scope == "evening":
                ev_mask = day_mask & np.isin(hours, list(evening_hours))
                mult[ev_mask] *= factor
    return mult


def festival_label_map():
    """date -> comma-joined labels (for reporting which dates were shaped)."""
    out = {}
    for d, _f, _s, label in FESTIVAL_EVENTS:
        out.setdefault(d, [])
        if label not in out[d]:
            out[d].append(label)
    return {d: ", ".join(v) for d, v in out.items()}
