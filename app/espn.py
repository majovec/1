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


CORE = "https://sports.core.api.espn.com/v2/sports/soccer/leagues"
_RESOLVED = {}
_SLUGS = None


def _request(url: str, params=None, tries: int = 3):
    """Vrací (HTTP status, JSON nebo None). Na chyby 4xx se neopakuje (je to chyba adresy, ne výpadek)."""
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code < 500:
                try:
                    return r.status_code, r.json()
                except ValueError:
                    return r.status_code, None
            last = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url} selhalo: {last}")


def list_slugs() -> list:
    """Všechny fotbalové ligy, které ESPN zná (jejich slugy)."""
    global _SLUGS
    if _SLUGS is not None:
        return _SLUGS
    out, page = [], 1
    try:
        while page <= 6:
            st, js = _request(CORE, {"limit": 500, "page": page})
            if st != 200 or not js:
                break
            for it in js.get("items", []):
                ref = it.get("$ref", "")
                if "/leagues/" in ref:
                    out.append(ref.split("/leagues/")[1].split("?")[0])
            if page >= js.get("pageCount", 1):
                break
            page += 1
    except RuntimeError as exc:
        print(f"VAROVÁNÍ: seznam lig ESPN se nepodařilo načíst ({exc})", file=sys.stderr)
    _SLUGS = out
    return out


def _works(slug: str) -> bool:
    st, _ = _request(f"{BASE}/{slug}/scoreboard")
    return st == 200


def resolve(slug: str):
    """Ověří, že slug existuje. Když ne, zkusí najít jiný slug stejné země. Vrací slug nebo None."""
    if slug in _RESOLVED:
        return _RESOLVED[slug]
    found = slug if _works(slug) else None
    if found is None:
        country = slug.split(".")[0]
        same = [x for x in list_slugs() if x.split(".")[0] == country]
        print(f"VAROVÁNÍ: ESPN slug '{slug}' nefunguje. Ligy ESPN pro '{country}': {same or 'žádné'}", file=sys.stderr)
        for cand in sorted(same, key=lambda x: (not x.endswith(".1"), x)):
            if cand != slug and _works(cand):
                found = cand
                print(f"Používám '{cand}' místo '{slug}'.")
                break
    if found is None:
        print(f"VAROVÁNÍ: pro '{slug}' ESPN žádná použitelná data nemá, liga se přeskakuje.", file=sys.stderr)
    _RESOLVED[slug] = found
    return found


def fetch_range(div: str, slug: str, start: pd.Timestamp, end: pd.Timestamp, step_days: int = 30) -> pd.DataFrame:
    slug = resolve(slug)
    if slug is None:
        return pd.DataFrame(columns=COLUMNS)
    frames, cur, fails = [], start.normalize(), 0
    while cur <= end and fails < 3:
        nxt = min(cur + pd.Timedelta(days=step_days - 1), end)
        params = {"dates": f"{cur:%Y%m%d}-{nxt:%Y%m%d}", "limit": 500}
        try:
            st, js = _request(f"{BASE}/{slug}/scoreboard", params)
        except RuntimeError as exc:
            print(f"VAROVÁNÍ: {exc}", file=sys.stderr)
            st, js = 0, None
        if st == 200 and js:
            fails = 0
            frames.append(parse_events(js, div))
        else:
            fails += 1
            print(f"VAROVÁNÍ: ESPN {slug} {params['dates']} vrátilo HTTP {st}", file=sys.stderr)
        cur = nxt + pd.Timedelta(days=1)
        time.sleep(0.2)
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["div", "date", "home", "away"])
