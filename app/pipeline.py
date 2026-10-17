"""Celý řetězec jedním příkazem: stáhnout data → backtest → předpovědi → vyhodnocení → JSON pro dashboard.

    python -m app.pipeline            # skutečná data z football-data.co.uk
    python -m app.pipeline --demo     # syntetická data, funguje offline
"""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from app import data, demo, evaluate, export, features, stack, tennis, tracker


def london_now() -> pd.Timestamp:
    try:
        from zoneinfo import ZoneInfo
        return pd.Timestamp(datetime.now(ZoneInfo("Europe/London")).replace(tzinfo=None))
    except Exception:  # noqa: BLE001
        return pd.Timestamp(datetime.utcnow())


def run(history: pd.DataFrame, fixtures: pd.DataFrame, now: pd.Timestamp, data_dir: Path, docs_dir: Path,
        leagues=None, half_life=None, min_train=None, edge=None, upcoming_days=None, log=print) -> dict:
    leagues = leagues or config.LEAGUES
    half_life = half_life or config.HALF_LIFE_DAYS
    min_train = min_train if min_train is not None else config.MIN_TRAIN_MATCHES
    edge = edge if edge is not None else config.EDGE_THRESHOLD
    upcoming_days = upcoming_days or config.UPCOMING_DAYS

    finished = history[history["hg"].notna() & history["ag"].notna()].copy()
    log(f"Historie: {len(finished)} dohraných zápasů")

    finished = features.build(finished)
    bt = evaluate.backtest(finished, min_train, half_life, log=log)
    learned, learn_report = stack.train(bt, evaluate.blend_probs)
    stack.save(learned, learn_report, Path(data_dir) / "stacker.json")
    for k, v in learn_report.items():
        log(f"Učení ({k}): " + ("zapnuto" if v["enabled"] else "vypnuto – " + v.get("reason", "")))
    bt_summary = evaluate.summarise(bt)
    log(bt_summary["verdict"])

    preds = tracker.predict_upcoming(finished, fixtures, now, upcoming_days, half_life, learned)
    log(f"Předpovězeno nadcházejících zápasů: {len(preds)}")
    log_path = Path(data_dir) / "predictions_log.csv"
    plog = tracker.update_log(tracker.load_log(log_path), preds, now)
    tracker.save_log(plog, log_path)
    settled, pending = tracker.settle(plog, finished, now)
    live = tracker.live_summary(settled, pending, now)
    log(f"Vyhodnoceno vlastních predikcí: {live['settled']}, čeká: {live['pending']}")

    summary = {
        "generated_at": now,
        "leagues": [{"code": c, "name": config.LEAGUE_NAMES.get(c, c)} for c in leagues],
        "market_weight": config.MARKET_WEIGHT,
        "history_matches": int(len(finished)),
        "history_from": finished["date"].min() if len(finished) else None,
        "history_to": finished["date"].max() if len(finished) else None,
        "config": {"half_life_days": half_life, "min_train_matches": min_train, "edge_threshold": edge,
                   "upcoming_days": upcoming_days},
        "backtest": bt_summary,
        "learning": learn_report,
        "live": live,
        "league_info": {c: dict(zip(("country", "flag", "name"), config.LEAGUE_INFO.get(c, (c, "", c)))) for c in leagues},
        "no_odds": [c for c in leagues if c in config.ESPN_LEAGUES],
    }
    docs_dir = Path(docs_dir)
    export.write_json(docs_dir / "summary.json", summary)
    export.write_json(docs_dir / "upcoming.json", {"generated_at": now, "matches": export.upcoming_rows(plog, now, edge)})
    export.write_json(docs_dir / "settled.json", {"generated_at": now, "matches": tracker.settled_rows(settled)})
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="syntetická data (offline)")
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    ap.add_argument("--docs-dir", default=str(config.DOCS_DATA_DIR))
    args = ap.parse_args(argv)

    now = london_now()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.demo:
        history, fixtures = demo.make_demo(now)
    else:
        history = data.fetch_history(config.LEAGUES, config.N_SEASONS, now, data_dir / "matches.csv")
        history.to_csv(data_dir / "matches.csv", index=False)
        fixtures = data.fetch_fixtures(config.LEAGUES, now, config.UPCOMING_DAYS)
        print(f"Nadcházející zápasy ve zdroji: {len(fixtures)}")
    run(history, fixtures, now, data_dir, Path(args.docs_dir))
    if not args.demo:
        try:
            tennis.run(now, data_dir, Path(args.docs_dir))
        except Exception as exc:  # noqa: BLE001  tenis nesmí shodit fotbal
            print(f"VAROVÁNÍ: tenis selhal: {exc}")


if __name__ == "__main__":
    main()
