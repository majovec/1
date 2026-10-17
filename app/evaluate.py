"""Chronologický backtest a metriky kvality (log loss, Brier, přesnost) včetně srovnání s kurzy."""
import numpy as np
import pandas as pd

import config
from app import features
from app.model import fit, predict

EPS = 1e-12
THRESHOLDS = (0.02, 0.05, 0.10)


def outcome_index(hg, ag):
    """0 = výhra domácích, 1 = remíza, 2 = výhra hostů."""
    hg, ag = np.asarray(hg), np.asarray(ag)
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def score(P, y):
    P, y = np.asarray(P, float), np.asarray(y, int)
    n = len(y)
    if n == 0:
        return None
    ll = -np.log(np.clip(P[np.arange(n), y], EPS, 1.0))
    brier = ((P - np.eye(3)[y]) ** 2).sum(axis=1)
    return {"n": int(n), "accuracy": float((P.argmax(axis=1) == y).mean()),
            "log_loss": float(ll.mean()), "brier": float(brier.mean())}


def implied_probs(odds):
    """Kurzy → pravděpodobnosti bez marže bookmakera."""
    inv = 1.0 / np.asarray(odds, float)
    return inv / inv.sum(axis=1, keepdims=True)


def blend_probs(P, O, w=None):
    """Smíchá model s trhem: w * (kurzy bez marže) + (1-w) * model. Trh je v praxi přesnější než samotný model."""
    w = config.MARKET_WEIGHT if w is None else w
    out = w * implied_probs(O) + (1 - w) * np.asarray(P, float)
    return out / out.sum(axis=1, keepdims=True)


def value_bets(P, O, y, thresholds=THRESHOLDS):
    """Plochá sázka 1 jednotky na výsledek s největší očekávanou výhodou p*kurz-1 >= práh."""
    ev = P * O - 1.0
    best, best_ev = ev.argmax(axis=1), ev.max(axis=1)
    rows = []
    for t in thresholds:
        sel = best_ev >= t
        k = int(sel.sum())
        if k == 0:
            rows.append({"threshold": t, "bets": 0, "hit_rate": None, "roi": None, "avg_odds": None})
            continue
        pick, odds = best[sel], O[sel, best[sel]]
        hit = pick == y[sel]
        profit = np.where(hit, odds - 1.0, -1.0)
        rows.append({"threshold": t, "bets": k, "hit_rate": float(hit.mean()),
                     "roi": float(profit.mean()), "avg_odds": float(odds.mean())})
    return rows


def backtest(finished: pd.DataFrame, min_train: int = 300, half_life: float = 450.0, log=print) -> pd.DataFrame:
    """Každý zápas se předpovídá jen z dřívějších dnů. Prvních `min_train` zápasů ligy slouží jen k učení."""
    rows = []
    if "hform" not in finished.columns:
        finished = features.build(finished)
    for div, g in finished.groupby("div"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) <= min_train:
            log(f"{div}: málo zápasů pro backtest ({len(g)} ≤ {min_train})")
            continue
        test = g.iloc[min_train:]
        for day, tg in test.groupby(test["date"].dt.normalize()):
            hist = g[g["date"] < day]
            model = fit(hist, day, half_life)
            freq = outcome_index(hist["hg"], hist["ag"])
            base = np.bincount(freq, minlength=3) / max(len(freq), 1)
            for r in tg.itertuples():
                p = predict(model, r.home, r.away)
                rows.append({
                    "div": div, "date": r.date, "home": r.home, "away": r.away,
                    "y": int(outcome_index(r.hg, r.ag)),
                    "p_h": p["p_home"], "p_d": p["p_draw"], "p_a": p["p_away"],
                    "b_h": base[0], "b_d": base[1], "b_a": base[2],
                    "o_h": r.odds_h, "o_d": r.odds_d, "o_a": r.odds_a,
                    "hform": r.hform, "aform": r.aform, "hrest": r.hrest, "arest": r.arest,
                    "p_over25": p["p_over25"], "p_btts": p["p_btts"],
                    "over25": int(r.hg + r.ag >= 3), "btts": int(r.hg > 0 and r.ag > 0),
                })
        log(f"{div}: otestováno {len(test)} zápasů")
    return pd.DataFrame(rows)


def calibration(P, y, bins=10):
    """Spolehlivost: předpovězená pravděpodobnost vs. skutečná četnost (všechny tři výsledky dohromady)."""
    p = np.asarray(P).ravel()
    hit = np.eye(3)[np.asarray(y)].ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        out.append({"predicted": float(p[sel].mean()), "observed": float(hit[sel].mean()), "n": int(sel.sum())})
    return out


def verdict(out: dict) -> str:
    n = out.get("matches_tested", 0)
    if n < 200:
        return "Zatím málo otestovaných zápasů (méně než 200), závěry by byly nespolehlivé."
    m, b = out["model"]["log_loss"], out["baseline"]["log_loss"]
    parts = []
    if m < b - 0.005:
        parts.append(f"Model má skutečný signál: log loss {m:.3f} je lepší než naivní odhad {b:.3f}.")
    else:
        parts.append(f"Model zatím není lepší než prostý odhad podle průměrné četnosti výsledků ({m:.3f} vs {b:.3f}).")
    mk = out.get("market")
    if mk:
        mm, k = out["model_on_market_subset"]["log_loss"], mk["log_loss"]
        if mm < k:
            parts.append(f"Na srovnatelných zápasech je lepší než kurzy bookmakerů ({mm:.3f} vs {k:.3f}). "
                         "To je velmi neobvyklé, ověř na dalších datech.")
        else:
            parts.append(f"Kurzy bookmakerů jsou přesnější ({k:.3f} vs {mm:.3f}). To je běžné; "
                         "sázky podle modelu by pravděpodobně prodělávaly.")
    return " ".join(parts)


def summarise(bt: pd.DataFrame) -> dict:
    if bt.empty:
        out = {"matches_tested": 0}
        out["verdict"] = verdict(out)
        return out
    P = bt[["p_h", "p_d", "p_a"]].to_numpy()
    B = bt[["b_h", "b_d", "b_a"]].to_numpy()
    y = bt["y"].to_numpy()
    out = {
        "matches_tested": int(len(bt)),
        "from": bt["date"].min().isoformat(), "to": bt["date"].max().isoformat(),
        "model": score(P, y), "baseline": score(B, y),
        "calibration": calibration(P, y),
        "over25": {"accuracy": float(((bt["p_over25"] >= 0.5).astype(int) == bt["over25"]).mean()),
                   "brier": float(((bt["p_over25"] - bt["over25"]) ** 2).mean()),
                   "base_rate": float(bt["over25"].mean())},
        "btts": {"accuracy": float(((bt["p_btts"] >= 0.5).astype(int) == bt["btts"]).mean()),
                 "brier": float(((bt["p_btts"] - bt["btts"]) ** 2).mean()),
                 "base_rate": float(bt["btts"].mean())},
        "by_league": {d: score(g[["p_h", "p_d", "p_a"]].to_numpy(), g["y"].to_numpy())
                      for d, g in bt.groupby("div")},
    }
    has = bt[["o_h", "o_d", "o_a"]].notna().all(axis=1).to_numpy()
    if has.sum() >= 30:
        O = bt.loc[has, ["o_h", "o_d", "o_a"]].to_numpy(float)
        out["market"] = score(implied_probs(O), y[has])
        out["model_on_market_subset"] = score(P[has], y[has])
        out["blend"] = score(blend_probs(P[has], O), y[has])
        out["value_bets"] = value_bets(blend_probs(P[has], O), O, y[has])
    out["verdict"] = verdict(out)
    return out
