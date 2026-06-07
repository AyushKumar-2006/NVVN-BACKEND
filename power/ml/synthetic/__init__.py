"""Hyper-realistic synthetic CG 5-min demand generation.

Backbone = an empirical (month, day-of-week, 5-min-slot) load profile learned
from the *real* StateLoad5Min CG data (2024-2026). That profile already embeds
the real seasonal, weekly, intraday, industrial (Bhilai/Korba/cement/mining) and
agricultural (irrigation) structure, so those patterns are *preserved* rather
than re-synthesised from scratch (which would double-count and break the
distribution match). On top of the backbone we overlay only what a smooth
climatological average cannot represent on a specific historical calendar:
festival effects, day-specific weather anomalies, smooth AR(1) noise + rare grid
events, and year-on-year growth.

See `generator.py` for the assembly pipeline and `events.py` for the CG festival
calendar.
"""
