"""ESPN veřejné JSON API (bez klíče): výsledky i rozpis pro ligy, které football-data.co.uk nemá (např. česká liga).

Časy z API jsou UTC; převádíme na britský čas, stejně jako zbytek projektu. Kurzy ESPN neposkytuje.
"""
import sys
import time

import numpy as np
import pandas as pd
import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
HEADERS = {"User-Agent": "football-ai-analyst/4.0 (GitHub Actions)"}
COLUMNS = ["div", "date", "home", "away", "hg", "ag", "odds_h", "odds_d", "odds_a"]


def parse_events(js: dict, div: str) -> pd.DataFrame:
    rows = []
    for ev in (js or {}).get("events", []) or []:
        try:
            comp = ev["competitions"][0]
            teams = {c["homeAway"]: c for c in comp["competitors"]}
            h, a = teams["home"], teams["away"]
            st = (comp.get("status") or ev.get("status") or {}).get("type", {})
            done = bool(st.get("completed")) and st.get("state") == "post"
            hg = pd.to_numeric(h.get("score"), errors="coerce") if done else np.nan
            ag = pd.to_numeric(a.get("score"), errors="coerce") if done else np.nan
            if done and (pd.isna(hg) or pd.isna(ag)):
                continue
            if not done and st.get("state") == "post":
                continue  # odložený / zrušený zápas
            when = pd.Timestamp(ev["date"])
            when = (when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC"))
            when = when.tz_convert("Europe/London").tz_localize(None)
            rows.append({"div": div, "date": when, "home": h["team"]["displayName"].strip(),
                         "away": a["team"]["displayName"].strip(), "hg": hg, "ag": ag,
                         "odds_h": np.nan, "odds_d": np.nan, "odds_a": np.nan})
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return pd.DataFrame(rows, columns=COLUMNS)


def _get(slug: str, start: pd.Timestamp, end: pd.Timestamp, tries: int = 3):
    url = f"{BASE}/{slug}/scoreboard"
    params = {"dates": f"{start:%Y%m%d}-{end:%Y%m%d}", "limit": 500}
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"ESPN {slug} {params['dates']} selhalo: {last}")


def fetch_range(div: str, slug: str, start: pd.Timestamp, end: pd.Timestamp, step_days: int = 60) -> pd.DataFrame:
    frames, cur = [], start.normalize()
    while cur <= end:
        nxt = min(cur + pd.Timedelta(days=step_days - 1), end)
        try:
            js = _get(slug, cur, nxt)
        except RuntimeError as exc:
            print(f"VAROVÁNÍ: {exc}", file=sys.stderr)
            js = None
        if js is not None:
            frames.append(parse_events(js, div))
        cur = nxt + pd.Timedelta(days=1)
        time.sleep(0.25)
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["div", "date", "home", "away"])
