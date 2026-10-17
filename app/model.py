"""Poissonův model: síla útoku/obrany týmů s časovým útlumem, zmenšením k průměru a Dixon–Coles korekcí."""
import math

import numpy as np
import pandas as pd

VERSION = "poisson-dc-decay-v3"
RHO = -0.08          # Dixon–Coles korekce nízkých skóre (zvyšuje 0:0 a 1:1)
MAX_GOALS = 10
DEFAULT_MU_H, DEFAULT_MU_A = 1.45, 1.15
_LOGFACT = np.array([math.lgamma(i + 1) for i in range(MAX_GOALS + 1)])


def fit(hist: pd.DataFrame, ref_date: pd.Timestamp, half_life: float = 450.0,
        prior: float = 8.0, iters: int = 30) -> dict:
    """Naučí model z `hist` (sloupce date, home, away, hg, ag) k datu `ref_date`.

    Novější zápasy mají větší váhu (poločas `half_life` dní). `prior` je počet „pseudo-gólů“,
    kterými se týmy s malou historií přitahují k průměru ligy.
    """
    n = len(hist)
    if n == 0:
        return {"mu_h": DEFAULT_MU_H, "mu_a": DEFAULT_MU_A, "att": {}, "dfn": {}, "games": {}, "n": 0,
                "version": VERSION}
    age = np.clip((ref_date - hist["date"]).dt.total_seconds().to_numpy() / 86400.0, 0, None)
    w = 0.5 ** (age / half_life)
    codes, teams = pd.factorize(pd.concat([hist["home"], hist["away"]], ignore_index=True))
    hi, ai = codes[:n], codes[n:]
    hg = hist["hg"].to_numpy(float)
    ag = hist["ag"].to_numpy(float)
    T = len(teams)
    att = np.ones(T)
    dfn = np.ones(T)
    for _ in range(iters):
        mu_h = (w * hg).sum() / (w * att[hi] * dfn[ai]).sum()
        mu_a = (w * ag).sum() / (w * att[ai] * dfn[hi]).sum()
        scored = np.bincount(hi, w * hg, T) + np.bincount(ai, w * ag, T)
        expected = np.bincount(hi, w * mu_h * dfn[ai], T) + np.bincount(ai, w * mu_a * dfn[hi], T)
        att = (scored + prior) / (expected + prior)
        conceded = np.bincount(hi, w * ag, T) + np.bincount(ai, w * hg, T)
        expected = np.bincount(hi, w * mu_a * att[ai], T) + np.bincount(ai, w * mu_h * att[hi], T)
        dfn = (conceded + prior) / (expected + prior)
        att = att / att.mean()
        dfn = dfn / dfn.mean()
    mu_h = (w * hg).sum() / (w * att[hi] * dfn[ai]).sum()
    mu_a = (w * ag).sum() / (w * att[ai] * dfn[hi]).sum()
    games = np.bincount(hi, minlength=T) + np.bincount(ai, minlength=T)
    names = list(teams)
    return {
        "mu_h": float(mu_h), "mu_a": float(mu_a),
        "att": {t: float(a) for t, a in zip(names, att)},
        "dfn": {t: float(d) for t, d in zip(names, dfn)},
        "games": {t: int(g) for t, g in zip(names, games)},
        "n": int(n), "version": VERSION,
    }


def score_matrix(lh: float, la: float, max_goals: int = MAX_GOALS, rho: float = RHO) -> np.ndarray:
    """Matice pravděpodobností přesných skóre; řádek = góly domácích, sloupec = góly hostů."""
    g = np.arange(max_goals + 1)
    ph = np.exp(-lh + g * np.log(lh) - _LOGFACT[: max_goals + 1])
    pa = np.exp(-la + g * np.log(la) - _LOGFACT[: max_goals + 1])
    m = np.outer(ph, pa)
    m[0, 0] *= 1 - lh * la * rho
    m[0, 1] *= 1 + lh * rho
    m[1, 0] *= 1 + la * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def predict(model: dict, home: str, away: str) -> dict:
    att, dfn = model["att"], model["dfn"]
    lh = model["mu_h"] * att.get(home, 1.0) * dfn.get(away, 1.0)
    la = model["mu_a"] * att.get(away, 1.0) * dfn.get(home, 1.0)
    lh, la = float(np.clip(lh, 0.2, 4.5)), float(np.clip(la, 0.2, 4.5))
    m = score_matrix(lh, la)
    g = np.arange(m.shape[0])
    total = np.add.outer(g, g)
    flat = sorted(((float(m[i, j]), i, j) for i in g for j in g), reverse=True)[:3]
    return {
        "xg_home": lh, "xg_away": la,
        "p_home": float(np.tril(m, -1).sum()),
        "p_draw": float(np.trace(m)),
        "p_away": float(np.triu(m, 1).sum()),
        "p_over25": float(m[total >= 3].sum()),
        "p_btts": float(m[1:, 1:].sum()),
        "top_scores": [{"score": f"{i}:{j}", "p": p} for p, i, j in flat],
        "home_n": model["games"].get(home, 0),
        "away_n": model["games"].get(away, 0),
    }
