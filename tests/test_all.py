import json

import numpy as np
import pandas as pd

from app import data, demo, evaluate, model, pipeline

NOW1 = pd.Timestamp("2026-10-17 12:00")


def _demo(now):
    return demo.make_demo(now)


def test_season_codes():
    assert data.season_codes(pd.Timestamp("2026-10-17"), 3) == ["2627", "2526", "2425"]
    assert data.season_codes(pd.Timestamp("2027-03-01"), 2) == ["2627", "2526"]


def test_parse_csv_mixed_formats():
    csv = ("Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,AvgH,AvgD,AvgA\n"
           "E0,15/08/2025,20:00,Liverpool,Bournemouth,4,2,1.3,5.5,9.5,1.28,5.9,10.1\n"
           "E0,16/08/25,,Arsenal,Chelsea,,,1.9,3.5,4.1,,,\n"
           ",,,,,,,,,,,,\n")
    df = data.parse_csv(csv.encode())
    assert len(df) == 2
    assert df.loc[0, "date"] == pd.Timestamp("2025-08-15 20:00")
    assert df.loc[1, "date"] == pd.Timestamp("2025-08-16 15:00")  # chybějící čas → 15:00
    assert df.loc[0, "odds_h"] == 1.28            # preferuje průměr trhu
    assert df.loc[1, "odds_h"] == 1.9             # jinak Bet365
    assert np.isnan(df.loc[1, "hg"])


def test_probabilities_are_valid():
    hist, fx = _demo(NOW1)
    m = model.fit(hist, NOW1)
    p = model.predict(m, fx.loc[0, "home"], fx.loc[0, "away"])
    assert abs(p["p_home"] + p["p_draw"] + p["p_away"] - 1) < 1e-9
    assert 0 <= p["p_over25"] <= 1 and 0 <= p["p_btts"] <= 1
    unknown = model.predict(m, "Nikdo", "Nikdo2")
    assert abs(unknown["p_home"] + unknown["p_draw"] + unknown["p_away"] - 1) < 1e-9


def test_backtest_never_uses_future():
    hist, _ = _demo(NOW1)
    bt = evaluate.backtest(hist, min_train=300, log=lambda *_: None)
    assert len(bt) == len(hist) - 300
    assert bt["date"].min() >= hist.sort_values("date").iloc[300]["date"]


def test_pipeline_end_to_end(tmp_path):
    data_dir, docs_dir = tmp_path / "data", tmp_path / "docs"
    h1, f1 = _demo(NOW1)
    s1 = pipeline.run(h1, f1, NOW1, data_dir, docs_dir, leagues=["E0"], log=lambda *_: None)
    bt = s1["backtest"]
    assert bt["matches_tested"] > 500
    assert bt["model"]["log_loss"] < bt["baseline"]["log_loss"]  # model má signál vůči naivnímu odhadu
    assert s1["live"]["settled"] == 0 and s1["live"]["pending"] > 0

    # O týden později: zápasy z minula jsou dohrané a predikce se vyhodnotí.
    now2 = NOW1 + pd.Timedelta(days=8)
    h2, f2 = _demo(now2)
    s2 = pipeline.run(h2, f2, now2, data_dir, docs_dir, leagues=["E0"], log=lambda *_: None)
    assert s2["live"]["settled"] >= 5
    up = json.loads((docs_dir / "upcoming.json").read_text(encoding="utf-8"))
    assert len(up["matches"]) > 0
    st = json.loads((docs_dir / "settled.json").read_text(encoding="utf-8"))
    assert len(st["matches"]) >= 5


def test_log_predictions_frozen_after_kickoff(tmp_path):
    from app import tracker
    data_dir, docs_dir = tmp_path / "d", tmp_path / "w"
    h1, f1 = _demo(NOW1)
    pipeline.run(h1, f1, NOW1, data_dir, docs_dir, leagues=["E0"], min_train=10**6, log=lambda *_: None)
    before = tracker.load_log(data_dir / "predictions_log.csv")
    now2 = NOW1 + pd.Timedelta(days=8)
    h2, f2 = _demo(now2)
    pipeline.run(h2, f2, now2, data_dir, docs_dir, leagues=["E0"], min_train=10**6, log=lambda *_: None)
    after = tracker.load_log(data_dir / "predictions_log.csv").set_index("key")
    played = before[before["date"] <= now2].set_index("key")
    assert len(played) > 0
    for key, row in played.iterrows():  # odehrané zápasy se už nepřepisují
        assert after.loc[key, "p_home"] == row["p_home"]


def test_espn_parse_and_blend():
    from app import espn
    js = {"events": [
        {"date": "2026-10-17T15:00Z", "competitions": [{"status": {"type": {"completed": True, "state": "post"}},
          "competitors": [{"homeAway": "home", "score": "2", "team": {"displayName": "Sparta"}},
                          {"homeAway": "away", "score": "1", "team": {"displayName": "Slavia"}}]}]},
        {"date": "2026-10-25T13:00Z", "competitions": [{"status": {"type": {"completed": False, "state": "pre"}},
          "competitors": [{"homeAway": "home", "score": "0", "team": {"displayName": "Plzeň"}},
                          {"homeAway": "away", "score": "0", "team": {"displayName": "Baník"}}]}]},
        {"bad": 1}]}
    df = espn.parse_events(js, "CZ1")
    assert len(df) == 2
    assert df.loc[0, "hg"] == 2 and df.loc[0, "date"] == pd.Timestamp("2026-10-17 16:00")  # UTC -> UK (BST)
    assert np.isnan(df.loc[1, "hg"])
    P = np.array([[0.5, 0.3, 0.2]])
    O = np.array([[1.8, 3.6, 4.5]])
    B = evaluate.blend_probs(P, O, 0.7)
    assert abs(B.sum() - 1) < 1e-9
    assert evaluate.blend_probs(P, O, 0.0)[0, 0] == 0.5


def test_stacker_learns_real_signal_and_ignores_noise():
    from app import stack
    rng = np.random.default_rng(1)
    n = 3000
    P = rng.dirichlet([6, 3, 4], n)
    hform, aform = rng.uniform(0.5, 2.5, n), rng.uniform(0.5, 2.5, n)
    nan = np.full(n, np.nan)
    def outcomes(effect):
        lh = np.log(P[:, 0] / P[:, 1]) + effect * (hform - aform)
        la = np.log(P[:, 2] / P[:, 1])
        z = np.exp(np.column_stack([lh, np.zeros(n), la]))
        z /= z.sum(axis=1, keepdims=True)
        return np.array([rng.choice(3, p=r) for r in z])
    X = stack.features(P, hform, aform, nan, nan)
    p, rep = stack.train_one(X, outcomes(1.2), False, P)
    assert rep["enabled"] and p is not None            # skutečný vliv formy se naučí
    p2, rep2 = stack.train_one(X, outcomes(0.0), False, P)
    assert not rep2["enabled"]                          # bez vlivu zůstane původní odhad


def test_features_use_only_past():
    from app import features
    d = pd.DataFrame({"div": "X", "date": pd.to_datetime(["2026-01-01", "2026-01-04", "2026-01-08", "2026-01-12", "2026-01-15"]),
                      "home": ["A", "A", "A", "A", "A"], "away": ["B", "B", "B", "B", "B"],
                      "hg": [1, 2, 0, 3, 1], "ag": [0, 0, 1, 1, 1]})
    f = features.build(d)
    assert np.isnan(f.loc[0, "hform"]) and f.loc[0, "hrest"] != f.loc[0, "hrest"]
    assert f.loc[3, "hform"] == (3 + 3 + 0) / 3        # jen zápasy před čtvrtým
    assert f.loc[3, "hrest"] == 4


def test_espn_slug_fallback_and_fail_fast(monkeypatch=None):
    from app import espn
    calls = []
    good = {"events": []}
    def fake(url, params=None, tries=3):
        calls.append((url, params))
        if "sports.core" in url:
            return 200, {"items": [{"$ref": "http://x/leagues/cze.2?lang=en"}, {"$ref": "http://x/leagues/cze.1?lang=en"}], "pageCount": 1}
        if "/bad.1/" in url or "/cze.9/" in url:
            return 400, None
        return 200, good
    orig = espn._request
    espn._request, espn._RESOLVED = fake, {}
    espn._SLUGS = None
    try:
        assert espn.resolve("cze.9") == "cze.1"            # špatný slug se nahradí funkčním ze stejné země
        assert espn.resolve("bad.1") is None
        before = len(calls)
        df = espn.fetch_range("X", "bad.1", pd.Timestamp("2020-01-01"), pd.Timestamp("2024-01-01"))
        assert df.empty and len(calls) == before            # neexistující liga se nestahuje měsíc po měsíci
    finally:
        espn._request = orig


def _tennis_hist(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    names = [f"Player{chr(65 + i)} {chr(65 + i)}." for i in range(30)] + ["De Minaur A.", "Auger-Aliassime F."]
    skill = rng.normal(0, 1.0, len(names))
    rows, d0 = [], pd.Timestamp("2024-01-01")
    for i in range(n):
        a, b = rng.choice(len(names), 2, replace=False)
        p = 1 / (1 + np.exp(-(skill[a] - skill[b])))
        w, l = (a, b) if rng.random() < p else (b, a)
        pm = 1 / (1 + np.exp(-(skill[w if w == a else a] - skill[b if w == a else b])))
        fair = 1 / (1 + np.exp(-(skill[w] - skill[l])))
        rows.append({"tour": "atp", "date": d0 + pd.Timedelta(hours=3 * i), "tournament": "Open X", "surface": "Hard",
                     "round": "1st", "w": names[w], "l": names[l],
                     "odds_w": round(1.05 / fair, 2), "odds_l": round(1.05 / (1 - fair), 2)})
    return pd.DataFrame(rows)


def test_tennis_end_to_end_and_names(tmp_path):
    from app import tennis
    hist = _tennis_hist()
    now = hist["date"].max() + pd.Timedelta(hours=1)
    raw = [{"tour": "atp", "date": now + pd.Timedelta(hours=5), "tournament": "Open X", "a": "Alex de Minaur", "b": "Felix Auger-Aliassime"},
           {"tour": "atp", "date": now + pd.Timedelta(hours=6), "tournament": "Open X", "a": "Neznámý Hráč", "b": "Felix Auger-Aliassime"}]
    s = tennis.run(now, tmp_path / "d", tmp_path / "w", hist=hist, raw_upcoming=raw, log=lambda *_: None)
    assert s["backtest"]["accuracy"] > 0.55                     # Elo se skutečně naučí sílu hráčů
    assert s["backtest"]["log_loss"] < s["backtest"]["naive_log_loss"]
    out = json.loads((tmp_path / "w" / "tennis.json").read_text(encoding="utf-8"))
    assert len(out["upcoming"]) == 1                            # neznámý hráč se přeskočí
    u = out["upcoming"][0]
    assert u["a"] == "De Minaur A." and u["b"] == "Auger-Aliassime F." and 0 < u["p_a"] < 1


def test_tennis_parse_frame_and_espn():
    from app import tennis
    raw = pd.DataFrame({"Date": ["2026-01-05", "2026-01-06"], "Tournament": ["Brisbane", "Brisbane"], "Surface": ["Hard"] * 2,
                        "Round": ["1st Round"] * 2, "Winner": ["Sinner J.", "Zverev A."], "Loser": ["Kyrgios N.", "Ruud C."],
                        "B365W": [1.2, 1.5], "B365L": [4.5, 2.6], "AvgW": [1.25, None], "AvgL": [4.2, None],
                        "Comment": ["Completed", "Retired"]})
    df = tennis.parse_frame(raw, "atp")
    assert len(df) == 1 and df.iloc[0]["odds_w"] == 1.25        # skreč se nepočítá, preferuje průměr trhu
    js = {"events": [{"name": "Brisbane", "groupings": [
        {"grouping": {"displayName": "Men's Singles"}, "competitions": [
            {"date": "2026-10-18T10:00Z", "status": {"type": {"state": "pre"}},
             "competitors": [{"athlete": {"displayName": "Jannik Sinner"}}, {"athlete": {"displayName": "Nick Kyrgios"}}]}]},
        {"grouping": {"displayName": "Men's Doubles"}, "competitions": [
            {"date": "2026-10-18T10:00Z", "status": {"type": {"state": "pre"}},
             "competitors": [{"athlete": {"displayName": "A B"}}, {"athlete": {"displayName": "C D"}}]}]}]}]}
    got = tennis.parse_espn_tennis(js, "atp")
    assert len(got) == 1 and got[0]["a"] == "Jannik Sinner"
    assert tennis.match_name("Jannik Sinner", tennis.build_index(df)) == "Sinner J."


def test_espn_mode_ladder_falls_back_to_single_day():
    from app import espn
    seen = []
    def fake(url, params=None, tries=3):
        seen.append(params)
        if "sports.core" in url:
            return 200, {"items": [], "pageCount": 1}
        d = (params or {}).get("dates", "")
        if "dates" not in (params or {}) or "-" not in d:   # jen jednodenní dotaz projde, rozsah = 400
            return 200, {"events": []}
        return 400, {"message": "bad"}
    orig = espn._request
    espn._request, espn._RESOLVED, espn._MODE = fake, {}, {}
    try:
        assert espn.mode_for("cze.1") == (1, False)
        df = espn.fetch_range("CZ1", "cze.1", pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-04"))
        assert df.empty
        assert sum(1 for p in seen if p and p.get("dates") == "20261002") == 1   # skutečně po dnech
    finally:
        espn._request = orig


def test_tennis_sackmann_parse_and_fullname_match():
    from app import tennis
    raw = pd.DataFrame({"tourney_date": [20260105, 20260105, 20260105], "tourney_name": ["Brisbane"] * 3, "surface": ["Hard"] * 3,
                        "round": ["R32", "F", "R16"], "winner_name": ["Jannik Sinner", "Jannik Sinner", "Alex de Minaur"],
                        "loser_name": ["Nick Kyrgios", "Carlos Alcaraz", "Casper Ruud"], "score": ["6-4 6-4", "6-3 RET", "6-1 6-2"]})
    df = tennis.parse_sackmann(raw, "atp")
    assert len(df) == 2                                         # skreč se nepočítá
    idx = tennis.build_index(df)
    assert tennis.match_name("Jannik Sinner", idx) == "Jannik Sinner"
    assert tennis.match_name("Alex De Minaur", idx) == "Alex de Minaur"
