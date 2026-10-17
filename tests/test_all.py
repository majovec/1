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
