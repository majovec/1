"""Bezplatný externí zdroj tenisových kurzů a historie jejich snímků.

Zdroj: veřejný matches.json projektu Mriganka-codes/tennis_data.
Neuvádí konkrétního bookmakera; aplikace proto zobrazuje externí tržní kurz,
nikoli nabídku Tipsportu/Fortuny. Staré zdrojové snímky se nepoužívají jako aktuální kurzy.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import requests

ODDS_URL = "https://raw.githubusercontent.com/Mriganka-codes/tennis_data/main/matches.json"
SOURCE_NAME = "Tennis Explorer (přes veřejný datový feed; bookmaker neuveden)"
MAX_AGE_HOURS = 12.0
HEADERS = {"User-Agent": "Football-AI-Analyst/6.8 (tennis odds importer)"}
SNAPSHOT_COLS = ["match_key", "snapshot_at", "source_updated_at", "match_date", "tour", "tournament",
                 "player_a", "player_b", "odds_a", "odds_b", "model_p_a", "market_p_a",
                 "overround", "source", "result_winner"]


def _utc(value) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    except Exception:
        return None


def _text(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    value = re.sub(r"\([^)]*\)", " ", value)  # seed/ranking e.g. (25)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _player_key(value: str) -> str:
    return _text(value)


def _tournament_score(a: str, b: str) -> float:
    aa, bb = _text(a), _text(b)
    if not aa or not bb:
        return 0.0
    if aa in bb or bb in aa:
        return 1.0
    ta, tb = set(aa.split()), set(bb.split())
    overlap = len(ta & tb) / max(1, len(ta | tb))
    return max(overlap, SequenceMatcher(None, aa, bb).ratio())


def _now_utc(value) -> pd.Timestamp | None:
    """The pipeline supplies naive Europe/London time; explicit timestamps keep their own zone."""
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("Europe/London")
        return ts.tz_convert("UTC")
    except Exception:
        return None


def parse_feed(payload: dict, now) -> dict:
    """Validate feed and return normalized matches; source timestamps without zone are UTC."""
    now_utc = _now_utc(now) or pd.Timestamp.now(tz="UTC")
    updated = _utc((payload or {}).get("last_updated"))
    if updated is None:
        return {"ok": False, "reason": "Zdroj neobsahuje platný čas aktualizace.", "matches": [], "updated_at": None, "age_hours": None}
    age = max(0.0, (now_utc - updated).total_seconds() / 3600.0)
    matches = []
    for item in (payload or {}).get("matches", []) or []:
        try:
            p1, p2 = str(item.get("player1", "")).strip(), str(item.get("player2", "")).strip()
            o1, o2 = float(item.get("odds1")), float(item.get("odds2"))
            if not p1 or not p2 or not (1.001 <= o1 <= 1000 and 1.001 <= o2 <= 1000):
                continue
            # The source publishes one day's card; its timestamp date is the only reliable date field.
            match_date = updated.normalize()
            clock = re.search(r"\b(\d{1,2}):(\d{2})\b", str(item.get("time", "")))
            if clock:
                match_date = match_date + pd.Timedelta(hours=int(clock.group(1)), minutes=int(clock.group(2)))
            matches.append({
                "tour": str(item.get("tour", "")).upper(),
                "date": match_date,
                "tournament": str(item.get("tournament", "")).strip(),
                "player1": p1, "player2": p2, "odds1": o1, "odds2": o2,
                "key_players": frozenset((_player_key(p1), _player_key(p2))),
            })
        except (TypeError, ValueError):
            continue
    if age > MAX_AGE_HOURS:
        return {"ok": False, "reason": f"Zdroj kurzů je zastaralý ({age:.1f} h; limit {MAX_AGE_HOURS:.0f} h).", "matches": [], "updated_at": updated, "age_hours": age}
    if not matches:
        return {"ok": False, "reason": "Zdroj neobsahuje žádné platné kurzy.", "matches": [], "updated_at": updated, "age_hours": age}
    return {"ok": True, "reason": "", "matches": matches, "updated_at": updated, "age_hours": age}


def fetch_feed(now, session=requests) -> dict:
    try:
        response = session.get(ODDS_URL, headers=HEADERS, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Odpověď zdroje není JSON objekt.")
        return parse_feed(payload, now)
    except Exception as exc:  # network errors must not break tennis predictions
        return {"ok": False, "reason": f"Zdroj kurzů není dostupný ({type(exc).__name__}).", "matches": [], "updated_at": None, "age_hours": None}


def _same_player(source_name: str, prediction_name: str) -> bool:
    # This source and the historical tennis data generally use surname + initials (e.g. Zverev A.).
    return _player_key(source_name) == _player_key(prediction_name)


def match_odds(predictions: list[dict], feed: dict, now) -> list[dict]:
    """Attach only unambiguous player+tour+date+tournament matches; odds orientation is corrected."""
    updated = feed.get("updated_at")
    for pred in predictions:
        pred.update({"odds_a": None, "odds_b": None, "odds_source": SOURCE_NAME,
                     "odds_updated_at": updated, "odds_age_hours": feed.get("age_hours"),
                     "odds_status": feed.get("reason") or "Kurz se nepodařilo bezpečně spárovat.",
                     "market_p_a": None, "market_p_b": None, "market_overround": None,
                     "edge_a": None, "edge_b": None})
        if not feed.get("ok"):
            continue
        candidates = []
        for match in feed["matches"]:
            if match["tour"] and pred.get("tour", "").upper() and match["tour"] not in (pred["tour"].upper(), "ATP" if pred["tour"].lower() == "atp" else "WTA"):
                continue
            a_is_1 = _same_player(match["player1"], pred["a"])
            b_is_2 = _same_player(match["player2"], pred["b"])
            a_is_2 = _same_player(match["player2"], pred["a"])
            b_is_1 = _same_player(match["player1"], pred["b"])
            if not ((a_is_1 and b_is_2) or (a_is_2 and b_is_1)):
                continue
            pred_date = _utc(pred.get("date"))
            match_date = match["date"]
            if pred_date is not None and abs((pred_date.normalize() - match_date.normalize()).total_seconds()) > 86400:
                continue
            score = _tournament_score(match["tournament"], pred.get("tournament", ""))
            if score < 0.25:
                continue
            candidates.append((score, match, a_is_1))
        candidates.sort(key=lambda x: x[0], reverse=True)
        if not candidates:
            pred["odds_status"] = "Kurz pro tento zápas není ve zdroji nebo se nepodařilo ověřit shodu."
            continue
        # If two similarly plausible records exist, do not guess.
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.08:
            pred["odds_status"] = "Nejednoznačná shoda zápasu; kurz nepřiřazen."
            continue
        score, match, a_is_1 = candidates[0]
        oa, ob = (match["odds1"], match["odds2"]) if a_is_1 else (match["odds2"], match["odds1"])
        market_a, market_b = 1.0 / oa, 1.0 / ob
        overround = market_a + market_b
        pred.update({"odds_a": oa, "odds_b": ob, "odds_status": "Dostupný externí tržní kurz",
                     "odds_match_tournament": match["tournament"],
                     "market_p_a": market_a / overround, "market_p_b": market_b / overround,
                     "market_overround": overround - 1.0,
                     "edge_a": float(pred["p_a"]) * oa - 1.0,
                     "edge_b": (1.0 - float(pred["p_a"])) * ob - 1.0})
    return predictions


def save_snapshots(path: Path, predictions: list[dict], now) -> pd.DataFrame:
    """Append a snapshot only when source timestamp or odds change; repeated identical runs are idempotent."""
    path = Path(path)
    if path.exists():
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame(columns=SNAPSHOT_COLS)
    else:
        old = pd.DataFrame(columns=SNAPSHOT_COLS)
    for col in SNAPSHOT_COLS:
        if col not in old:
            old[col] = None
    rows = []
    now_utc = _now_utc(now) or pd.Timestamp.now(tz="UTC")
    for p in predictions:
        if p.get("odds_a") is None or p.get("odds_b") is None:
            continue
        key = f"{p.get('tour')}|{p.get('date')}|{'|'.join(sorted([str(p.get('a')), str(p.get('b'))]))}"
        same = old[(old.match_key.astype(str) == key) &
                   (old.source_updated_at.astype(str) == str(p.get("odds_updated_at"))) &
                   (pd.to_numeric(old.odds_a, errors="coerce") == float(p["odds_a"])) &
                   (pd.to_numeric(old.odds_b, errors="coerce") == float(p["odds_b"]))]
        if len(same):
            continue
        rows.append({"match_key": key, "snapshot_at": now_utc.isoformat(),
                     "source_updated_at": p.get("odds_updated_at"), "match_date": p.get("date"),
                     "tour": p.get("tour"), "tournament": p.get("tournament"),
                     "player_a": p.get("a"), "player_b": p.get("b"),
                     "odds_a": p.get("odds_a"), "odds_b": p.get("odds_b"),
                     "model_p_a": p.get("p_a"), "market_p_a": p.get("market_p_a"),
                     "overround": p.get("market_overround"), "source": SOURCE_NAME, "result_winner": ""})
    if rows:
        old = pd.concat([old[SNAPSHOT_COLS], pd.DataFrame(rows)], ignore_index=True)
    else:
        old = old[SNAPSHOT_COLS]
    path.parent.mkdir(parents=True, exist_ok=True)
    old.to_csv(path, index=False)
    return old


def settle_snapshots(path: Path, hist: pd.DataFrame) -> pd.DataFrame:
    """Fill result_winner once historical results contain the corresponding completed match."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLS)
    try:
        log = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=SNAPSHOT_COLS)
    if log.empty or hist is None or hist.empty:
        return log
    for col in SNAPSHOT_COLS:
        if col not in log:
            log[col] = ""
    for i, row in log[log["result_winner"].isna() | (log["result_winner"].astype(str).str.strip() == "")].iterrows():
        try:
            date = pd.Timestamp(row["match_date"])
            tour = str(row["tour"]).lower()
            a, b = str(row["player_a"]), str(row["player_b"])
            candidates = hist[(hist["tour"].astype(str).str.lower() == tour) &
                              (pd.to_datetime(hist["date"]) >= date.normalize() - pd.Timedelta(days=1)) &
                              (pd.to_datetime(hist["date"]) <= date.normalize() + pd.Timedelta(days=4)) &
                              (((hist["w"].astype(str) == a) & (hist["l"].astype(str) == b)) |
                               ((hist["w"].astype(str) == b) & (hist["l"].astype(str) == a)))]
            if len(candidates) == 1:
                log.at[i, "result_winner"] = str(candidates.iloc[0]["w"])
        except Exception:
            continue
    log[SNAPSHOT_COLS].to_csv(path, index=False)
    return log[SNAPSHOT_COLS]
