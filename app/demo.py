"""Syntetická data pro vyzkoušení celého řetězce offline (python -m app.pipeline --demo).

Týmy mají skryté síly, góly se losují z Poissonova rozdělení a kurzy jsou skutečné pravděpodobnosti
s drobným šumem a 6% marží. Slouží jen k testům, ne k poznání reálné kvality modelu.
"""
import numpy as np
import pandas as pd

from app.model import score_matrix


def _round_robin(n: int) -> list:
    teams = list(range(n))
    rounds = []
    for r in range(n - 1):
        pairs = [(teams[i], teams[n - 1 - i]) for i in range(n // 2)]
        rounds.append([(a, b) if (r + i) % 2 == 0 else (b, a) for i, (a, b) in enumerate(pairs)])
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds + [[(b, a) for a, b in rd] for rd in rounds]


def make_demo(now: pd.Timestamp, seed: int = 7, n_seasons: int = 4, n_teams: int = 20, days_ahead: int = 10):
    rng = np.random.default_rng(seed)
    names = [f"Demo {i:02d}" for i in range(1, n_teams + 1)]
    att = rng.normal(0, 0.25, n_teams)
    dfn = rng.normal(0, 0.25, n_teams)
    att -= att.mean()
    dfn -= dfn.mean()
    start_year = now.year if now.month >= 7 else now.year - 1
    rounds = _round_robin(n_teams)
    rows = []
    for s in range(n_seasons - 1, -1, -1):
        season_start = pd.Timestamp(year=start_year - s, month=8, day=10)
        for r, games in enumerate(rounds):
            for h, a in games:
                kick = (season_start + pd.Timedelta(days=7 * r + int(rng.integers(0, 3)))
                        + pd.Timedelta(hours=int(rng.choice([12, 15, 17, 20]))))
                lh = 1.40 * np.exp(att[h] + dfn[a])
                la = 1.10 * np.exp(att[a] + dfn[h])
                m = score_matrix(lh, la, rho=0.0)
                p = np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
                mk = p * np.exp(rng.normal(0, 0.04, 3))
                mk /= mk.sum()
                odds = np.round(1.0 / (mk * 1.06), 2)
                rows.append({"div": "E0", "date": kick, "home": names[h], "away": names[a],
                             "hg": float(rng.poisson(lh)), "ag": float(rng.poisson(la)),
                             "odds_h": odds[0], "odds_d": odds[1], "odds_a": odds[2]})
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    history = df[df["date"] <= now].reset_index(drop=True)
    fixtures = df[(df["date"] > now) & (df["date"] <= now + pd.Timedelta(days=days_ahead))].copy()
    fixtures["hg"] = np.nan
    fixtures["ag"] = np.nan
    return history, fixtures.reset_index(drop=True)
