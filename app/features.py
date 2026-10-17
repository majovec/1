"""Dodatečné ukazatele zápasu: forma (body na zápas z posledních 5) a odpočinek (dny od posledního zápasu).

Všechno se počítá jen z dřívějších zápasů, takže backtest nevidí budoucnost.
"""
from collections import deque

import numpy as np
import pandas as pd

FORM_N = 5


def _pts(a, b):
    return 3 if a > b else (1 if a == b else 0)


def _walk(g):
    """Projde zápasy jedné ligy chronologicky; vrací hodnoty PŘED zápasem a konečný stav týmů."""
    form, last = {}, {}
    cols = {k: [] for k in ("hform", "aform", "hrest", "arest")}
    for r in g.itertuples():
        for t, f, rs in ((r.home, "hform", "hrest"), (r.away, "aform", "arest")):
            d = form.get(t)
            cols[f].append(float(np.mean(d)) if d is not None and len(d) >= 3 else np.nan)
            cols[rs].append((r.date - last[t]).total_seconds() / 86400 if t in last else np.nan)
        form.setdefault(r.home, deque(maxlen=FORM_N)).append(_pts(r.hg, r.ag))
        form.setdefault(r.away, deque(maxlen=FORM_N)).append(_pts(r.ag, r.hg))
        last[r.home] = last[r.away] = r.date
    return cols, form, last


def build(finished: pd.DataFrame) -> pd.DataFrame:
    """Přidá sloupce hform, aform, hrest, arest (NaN, když je málo historie)."""
    parts = []
    for _, g in finished.sort_values("date").groupby("div", sort=False):
        cols, _, _ = _walk(g)
        parts.append(g.assign(**cols))
    return pd.concat(parts).sort_values("date").reset_index(drop=True) if parts else finished


def state(finished_div: pd.DataFrame) -> dict:
    """Aktuální stav týmů jedné ligy: {tým: (body/zápas z posledních 5 nebo NaN, datum posledního zápasu)}."""
    g = finished_div.sort_values("date")
    _, form, last = _walk(g)
    return {t: (float(np.mean(d)) if len(d) >= 3 else np.nan, last[t]) for t, d in form.items()}


def for_fixture(st: dict, home: str, away: str, when: pd.Timestamp) -> tuple:
    """(forma domácích, forma hostů, odpočinek domácích, odpočinek hostů ve dnech)."""
    def one(t):
        f, d = st.get(t, (np.nan, None))
        return f, (when - d).total_seconds() / 86400 if d is not None else np.nan
    (hf, hr), (af, ar) = one(home), one(away)
    return hf, af, hr, ar
