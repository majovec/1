"""Zápis JSON souborů pro statický dashboard (docs/data/*.json)."""
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.evaluate import implied_probs


def clean(o):
    """Převede numpy/pandas typy na čisté JSON (NaN → null)."""
    if o is pd.NaT:
        return None
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return o


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1), encoding="utf-8")


def upcoming_rows(log: pd.DataFrame, now: pd.Timestamp, edge_threshold: float) -> list:
    rows = []
    if log.empty:
        return rows
    up = log[log["date"] > now].sort_values("date")
    for r in up.itertuples():
        p = np.array([r.p_home, r.p_draw, r.p_away])
        odds = np.array([r.odds_h, r.odds_d, r.odds_a], float)
        item = {
            "div": r.div, "date": r.date, "home": r.home, "away": r.away,
            "p": p, "xg": [r.xg_home, r.xg_away], "over25": r.p_over25, "btts": r.p_btts,
            "top_scores": json.loads(r.top_scores) if isinstance(r.top_scores, str) else [],
            "low_confidence": bool(min(r.home_n, r.away_n) < 15),
            "odds": None, "market": None, "edge": None, "best": None,
        }
        if not np.isnan(odds).any():
            edge = p * odds - 1.0
            k = int(edge.argmax())
            item.update(odds=odds, market=implied_probs(odds.reshape(1, 3))[0], edge=edge,
                        best={"pick": k, "edge": float(edge[k]), "odds": float(odds[k]),
                              "flag": bool(edge[k] >= edge_threshold)})
        rows.append(item)
    return rows
