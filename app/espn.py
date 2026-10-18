"""ESPN veřejné JSON API (bez klíče): výsledky i rozpis pro ligy, které football-data.co.uk nemá (např. česká liga).

Časy z API jsou UTC; převádíme na britský čas, stejně jako zbytek projektu. Kurzy ESPN neposkytuje.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

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
_MODE = {}
LADDER = [(30, True), (30, False), (7, False), (1, False)]  # (délka okna ve dnech, posílat limit?)
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


def _params(cur, nxt, use_limit):
    p = {"dates": f"{cur:%Y%m%d}" if cur == nxt else f"{cur:%Y%m%d}-{nxt:%Y%m%d}"}
    if use_limit:
        p["limit"] = 500
    return p


def mode_for(slug: str, base: str = None):
    """ESPN u některých lig odmítá rozsah dat (HTTP 400). Najde první variantu, která projde:
    30denní okno → bez limitu → týdenní okna → po jednom dni."""
    base = base or BASE
    key = (base, slug)
    if key in _MODE:
        return _MODE[key]
    probe_end = pd.Timestamp.now().normalize() - pd.Timedelta(days=40)
    chosen = None
    for step, lim in LADDER:
        cur = probe_end - pd.Timedelta(days=step - 1)
        try:
            st, js = _request(f"{base}/{slug}/scoreboard", _params(cur, probe_end, lim))
        except RuntimeError:
            continue
        if st == 200:
            chosen = (step, lim)
            break
        print(f"ESPN {slug}: okno {step} dní{' s limitem' if lim else ''} → HTTP {st} {str(js)[:120] if js else ''}", file=sys.stderr)
    print(f"ESPN {slug}: použitý režim {chosen}")
    _MODE[key] = chosen
    return chosen


def _fetch_days(div: str, slug: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Po jednom dni, paralelně (8 vláken). U delší historie nejdřív zkontroluje posledních 75 dní:
    když tam ESPN nevrátí žádný zápas, liga se přeskočí (ušetří stovky zbytečných dotazů)."""
    end = end.normalize()
    if (end - start).days > 900:
        start = end - pd.Timedelta(days=900)
    days = list(pd.date_range(start, end))
    url = f"{BASE}/{slug}/scoreboard"

    def one(d):
        try:
            st, js = _request(url, {"dates": f"{d:%Y%m%d}"}, tries=2)
        except RuntimeError:
            return d, 0, None
        return d, st, js

    def run(ds):
        with ThreadPoolExecutor(max_workers=8) as ex:
            return list(ex.map(one, ds))

    results = []
    if len(days) > 100:
        probe = days[-75:]
        results = run(probe)
        got = sum(len((js or {}).get("events", []) or []) for _, st, js in results if st == 200)
        ok = sum(1 for _, st, _ in results if st == 200)
        if got == 0:
            print(f"VAROVÁNÍ: ESPN {slug}: za posledních 75 dní žádný zápas (úspěšných dotazů {ok}/75), liga se přeskakuje.", file=sys.stderr)
            return pd.DataFrame(columns=COLUMNS)
        days = days[:-75]
    results += run(days)
    bad = sum(1 for _, st, _ in results if st != 200)
    if bad:
        print(f"VAROVÁNÍ: ESPN {slug}: {bad} z {len(results)} dní se nepodařilo stáhnout.", file=sys.stderr)
    frames = [parse_events(js, div) for _, st, js in results if st == 200 and js]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["div", "date", "home", "away"])


def fetch_range(div: str, slug: str, start: pd.Timestamp, end: pd.Timestamp, step_days: int = None) -> pd.DataFrame:
    slug = resolve(slug)
    if slug is None:
        return pd.DataFrame(columns=COLUMNS)
    mode = mode_for(slug)
    if mode is None:
        print(f"VAROVÁNÍ: ESPN {slug} nepřijímá žádný formát dat, liga se přeskakuje.", file=sys.stderr)
        return pd.DataFrame(columns=COLUMNS)
    step, lim = mode
    start = start.normalize()
    if step == 1:
        return _fetch_days(div, slug, start, end)
    frames, cur, fails = [], start, 0
    while cur <= end and fails < 3:
        nxt = min(cur + pd.Timedelta(days=step - 1), end.normalize())
        try:
            st, js = _request(f"{BASE}/{slug}/scoreboard", _params(cur, nxt, lim))
        except RuntimeError as exc:
            print(f"VAROVÁNÍ: {exc}", file=sys.stderr)
            st, js = 0, None
        if st == 200 and js:
            fails = 0
            frames.append(parse_events(js, div))
        else:
            fails += 1
            print(f"VAROVÁNÍ: ESPN {slug} {cur:%Y%m%d}-{nxt:%Y%m%d} vrátilo HTTP {st}", file=sys.stderr)
        cur = nxt + pd.Timedelta(days=1)
        time.sleep(0.12 if step == 1 else 0.25)
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["div", "date", "home", "away"])
