"""Ukládání vlastních předpovědí do repozitáře a jejich vyhodnocení po odehrání zápasů."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.evaluate import EPS, THRESHOLDS, implied_probs, outcome_index, score, value_bets
from app.model import fit, predict

LOG_COLUMNS = [
    "key", "div", "date", "home", "away",
    "p_home", "p_draw", "p_away", "p_over25", "p_btts", "xg_home", "xg_away", "top_scores",
    "home_n", "away_n", "odds_h", "odds_d", "odds_a", "created_at", "updated_at",
]
DATE_COLS = ["date", "created_at", "updated_at"]


def season_year(d: pd.Timestamp) -> int:
    return d.year if d.month >= 7 else d.year - 1


def load_log(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(path, parse_dates=DATE_COLS)


def save_log(log: pd.DataFrame, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    log[LOG_COLUMNS].to_csv(path, index=False)


def predict_upcoming(finished: pd.DataFrame, fixtures: pd.DataFrame, now: pd.Timestamp,
                     days: int, half_life: float) -> pd.DataFrame:
    """Předpověď všech zápasů, které začnou v příštích `days` dnech (model učený k okamžiku `now`)."""
    up = fixtures[(fixtures["date"] > now) & (fixtures["date"] <= now + pd.Timedelta(days=days))]
    rows = []
    for div, f in up.groupby("div"):
        hist = finished[finished["div"] == div]
        if hist.empty:
            continue
        model = fit(hist, now, half_life)
        for r in f.itertuples():
            p = predict(model, r.home, r.away)
            rows.append({
                "key": f"{div}|{r.home}|{r.away}|{season_year(r.date)}",
                "div": div, "date": r.date, "home": r.home, "away": r.away,
                "p_home": p["p_home"], "p_draw": p["p_draw"], "p_away": p["p_away"],
                "p_over25": p["p_over25"], "p_btts": p["p_btts"],
                "xg_home": p["xg_home"], "xg_away": p["xg_away"],
                "top_scores": json.dumps(p["top_scores"]),
                "home_n": p["home_n"], "away_n": p["away_n"],
                "odds_h": r.odds_h, "odds_d": r.odds_d, "odds_a": r.odds_a,
            })
    return pd.DataFrame(rows)


def update_log(log: pd.DataFrame, preds: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Nové zápasy přidá, u dosud neodehraných přepíše predikci novější. Odehrané zůstávají zmrazené
    (nové predikce vznikají jen pro zápasy v budoucnosti), takže se výsledek nedá „dodatečně vylepšit“."""
    if preds.empty:
        return log
    if log.empty:
        out = preds.copy()
        out["created_at"] = now
        out["updated_at"] = now
        return out[LOG_COLUMNS].sort_values("date").reset_index(drop=True)
    log = log.copy()
    index = {k: i for i, k in enumerate(log["key"])}
    new_rows = []
    for r in preds.to_dict("records"):
        i = index.get(r["key"])
        if i is None:
            new_rows.append(dict(r, created_at=now, updated_at=now))
            continue
        for c in preds.columns:
            log.at[i, c] = r[c]
        log.at[i, "updated_at"] = now
    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
    return log[LOG_COLUMNS].sort_values("date").reset_index(drop=True)


def settle(log: pd.DataFrame, finished: pd.DataFrame, now: pd.Timestamp):
    """Spáruje uložené predikce se skutečnými výsledky. Vrací (vyhodnocené, čekající)."""
    if log.empty:
        return pd.DataFrame(), log
    f = finished[["div", "home", "away", "date", "hg", "ag"]].rename(columns={"date": "res_date"})
    m = log.merge(f, on=["div", "home", "away"], how="inner")
    m["gap"] = (m["res_date"] - m["date"]).abs()
    m = m[m["gap"] <= pd.Timedelta(days=5)].sort_values("gap").drop_duplicates("key")
    pending = log[~log["key"].isin(m["key"])]
    if m.empty:
        return pd.DataFrame(), pending
    m = m.sort_values("date").reset_index(drop=True)
    m["y"] = outcome_index(m["hg"], m["ag"])
    m["over25"] = ((m["hg"] + m["ag"]) >= 3).astype(int)
    m["btts"] = ((m["hg"] > 0) & (m["ag"] > 0)).astype(int)
    return m, pending


def live_summary(settled: pd.DataFrame, pending: pd.DataFrame, now: pd.Timestamp) -> dict:
    awaiting = int((pending["date"] <= now).sum()) if len(pending) else 0
    out = {"settled": int(len(settled)), "pending": int(len(pending)), "awaiting_result": awaiting}
    if settled.empty:
        return out
    P = settled[["p_home", "p_draw", "p_away"]].to_numpy(float)
    y = settled["y"].to_numpy()
    out["model"] = score(P, y)
    has = settled[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1).to_numpy()
    if has.sum() >= 5:
        O = settled.loc[has, ["odds_h", "odds_d", "odds_a"]].to_numpy(float)
        out["market"] = score(implied_probs(O), y[has])
        out["model_on_market_subset"] = score(P[has], y[has])
        out["value_bets"] = value_bets(P[has], O, y[has])
    return out


def settled_rows(settled: pd.DataFrame, limit: int = 150) -> list:
    rows = []
    if settled.empty:
        return rows
    for r in settled.sort_values("date", ascending=False).head(limit).itertuples():
        p = [r.p_home, r.p_draw, r.p_away]
        pick = int(np.argmax(p))
        rows.append({
            "div": r.div, "date": r.date, "home": r.home, "away": r.away,
            "score": f"{int(r.hg)}:{int(r.ag)}", "p": p, "pick": pick, "actual": int(r.y),
            "correct": pick == int(r.y), "odds": [r.odds_h, r.odds_d, r.odds_a],
        })
    return rows
