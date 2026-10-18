"""TheSportsDB (bezplatné veřejné API, klíč '123'): česká liga a další, které nemá ESPN ani football-data.co.uk.

Limit zdarma je ~30 dotazů za minutu, proto se mezi dotazy čeká. Kurzy API nemá.
Postup: najít ID ligy podle země → projít kola sezóny (eventsround) → plus dalších 15 zápasů (eventsnextleague).
"""
import sys
import time

import numpy as np
import pandas as pd
import requests

BASE = "https://www.thesportsdb.com/api/v1/json/123"
HEADERS = {"User-Agent": "Mozilla/5.0 football-ai-analyst"}
COLUMNS = ["div", "date", "home", "away", "hg", "ag", "odds_h", "odds_d", "odds_a"]
PAUSE = 2.2
_IDS = {}
BAD_WORDS = ("cup", "women", "second", "2.", " 2", "u19", "u21", "u23", "reserve", "playoff", "super cup", "division 2")


def _get(path: str, params: dict = None):
    for attempt in range(3):
        try:
            time.sleep(PAUSE)
            r = requests.get(f"{BASE}/{path}", params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                time.sleep(25)
                continue
            if r.status_code != 200:
                print(f"  TheSportsDB {path} {params} → HTTP {r.status_code}", file=sys.stderr)
                return None
            return r.json()
        except Exception as exc:  # noqa: BLE001
            print(f"  TheSportsDB {path}: {type(exc).__name__}: {str(exc)[:80]}", file=sys.stderr)
    return None


def pick_league(js, keywords) -> tuple:
    items = []
    for key in ("countries", "countrys", "leagues"):
        items += (js or {}).get(key) or []
    soccer = [x for x in items if str(x.get("strSport", "Soccer")).lower() == "soccer"]
    for kw in keywords:
        for x in soccer:
            if kw.lower() in str(x.get("strLeague", "")).lower() and not any(b in str(x["strLeague"]).lower() for b in ("cup", "women", "u19", "u21", "u23")):
                return x["idLeague"], x["strLeague"], [y.get("strLeague") for y in soccer]
    ok = [x for x in soccer if not any(b in str(x.get("strLeague", "")).lower() for b in BAD_WORDS)]
    if ok:
        return ok[0]["idLeague"], ok[0]["strLeague"], [y.get("strLeague") for y in soccer]
    return None, None, [y.get("strLeague") for y in soccer]


def league_id(div: str, country: str, keywords):
    if div in _IDS:
        return _IDS[div]
    js = _get("search_all_leagues.php", {"c": country, "s": "Soccer"})
    lid, name, seen = pick_league(js, keywords)
    print(f"TheSportsDB {div}: ligy v zemi {country}: {seen} → použita: {name} (id {lid})")
    _IDS[div] = lid
    return lid


def parse_events(js, div: str, now: pd.Timestamp = None) -> pd.DataFrame:
    rows = []
    now = now if now is not None else pd.Timestamp.now()
    for ev in (js or {}).get("events") or (js or {}).get("results") or []:
        try:
            home, away = str(ev.get("strHomeTeam") or "").strip(), str(ev.get("strAwayTeam") or "").strip()
            if not home or not away:
                continue
            status = str(ev.get("strStatus") or "").lower()
            if str(ev.get("strPostponed") or "").lower() == "yes" or any(x in status for x in ("postp", "cancel", "abandon", "susp")):
                continue
            ts = ev.get("strTimestamp") or (f"{ev.get('dateEvent')}T{ev.get('strTime') or '12:00:00'}" if ev.get("dateEvent") else None)
            if not ts:
                continue
            when = pd.Timestamp(ts)
            when = when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
            when = when.tz_convert("Europe/London").tz_localize(None)
            hs, as_ = pd.to_numeric(ev.get("intHomeScore"), errors="coerce"), pd.to_numeric(ev.get("intAwayScore"), errors="coerce")
            done = not (pd.isna(hs) or pd.isna(as_)) and when < now
            if not done and when < now:
                continue                                    # odehráno, ale bez výsledku: nevíme
            rows.append({"div": div, "date": when, "home": home, "away": away, "hg": hs if done else np.nan,
                         "ag": as_ if done else np.nan, "odds_h": np.nan, "odds_d": np.nan, "odds_a": np.nan})
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame(rows, columns=COLUMNS)


def season_labels(now: pd.Timestamp, n: int):
    start = now.year if now.month >= 7 else now.year - 1
    return [f"{start - i}-{start - i + 1}" for i in range(n)]


def fetch_season(lid, season: str, div: str, now: pd.Timestamp) -> pd.DataFrame:
    frames, empty = [], 0
    for r in range(1, 45):
        js = _get("eventsround.php", {"id": lid, "r": r, "s": season})
        df = parse_events(js, div, now)
        if len(df):
            frames.append(df)
            empty = 0
        else:
            empty += 1
            if empty >= 3 and (frames or r >= 3):
                break
    if not frames:                         # záloha: celá sezóna najednou (zdarma max. 100 zápasů)
        js = _get("eventsseason.php", {"id": lid, "s": season})
        df = parse_events(js, div, now)
        if len(df):
            print(f"TheSportsDB {div} {season}: kola nevrátila nic, použit eventsseason ({len(df)} zápasů)")
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return out.drop_duplicates(["div", "date", "home", "away"])


def fetch_history(div: str, country: str, keywords, now: pd.Timestamp, n_seasons: int, have_cache: bool) -> pd.DataFrame:
    lid = league_id(div, country, keywords)
    if not lid:
        print(f"VAROVÁNÍ: TheSportsDB pro {div} nenašlo ligu.", file=sys.stderr)
        return pd.DataFrame(columns=COLUMNS)
    seasons = season_labels(now, 1 if have_cache else min(n_seasons, 3))
    frames = [fetch_season(lid, s, div, now) for s in seasons]
    frames = [f for f in frames if len(f)]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    df = df[df["hg"].notna()]
    print(f"{div} (TheSportsDB): {len(df)} odehraných zápasů ze sezón {seasons}")
    return df


def fetch_upcoming(div: str, country: str, keywords, now: pd.Timestamp, days: int) -> pd.DataFrame:
    lid = league_id(div, country, keywords)
    if not lid:
        return pd.DataFrame(columns=COLUMNS)
    df = parse_events(_get("eventsnextleague.php", {"id": lid}), div, now)
    if len(df):
        df = df[(df["date"] > now) & (df["date"] <= now + pd.Timedelta(days=days + 1))]
    print(f"{div} (TheSportsDB): {len(df)} nadcházejících zápasů")
    return df
