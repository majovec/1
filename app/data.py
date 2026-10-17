"""Stahování dat z football-data.co.uk (zdarma, bez klíče): výsledky + kurzy + nadcházející zápasy.

Všechny časy jsou britský místní čas (tak je publikuje zdroj); ukládají se bez časové zóny.
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://www.football-data.co.uk"
HEADERS = {"User-Agent": "football-ai-analyst/3.0 (GitHub Actions)"}
COLUMNS = ["div", "date", "home", "away", "hg", "ag", "odds_h", "odds_d", "odds_a"]

# Sady kurzů v pořadí podle preference (průměr trhu je nejlepší).
ODDS_SETS = [
    ("AvgH", "AvgD", "AvgA"),
    ("B365H", "B365D", "B365A"),
    ("PSH", "PSD", "PSA"),
    ("BbAvH", "BbAvD", "BbAvA"),
    ("WHH", "WHD", "WHA"),
]


def season_codes(today: pd.Timestamp, n: int) -> list:
    """['2627', '2526', ...] od aktuální sezóny zpět. Sezóna začíná v červenci."""
    start = today.year if today.month >= 7 else today.year - 1
    return [f"{(start - k) % 100:02d}{(start - k + 1) % 100:02d}" for k in range(n)]


def _download(url: str, tries: int = 3):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=40)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"Stažení {url} selhalo: {last}")


def parse_csv(content: bytes, div_fallback: str = "") -> pd.DataFrame:
    """Převede CSV z football-data.co.uk na jednotný formát (COLUMNS)."""
    text = content.decode("utf-8-sig", errors="replace")
    raw = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
    need = {"Date", "HomeTeam", "AwayTeam"}
    if not need <= set(raw.columns):
        return pd.DataFrame(columns=COLUMNS)
    raw = raw.dropna(subset=list(need)).reset_index(drop=True)

    out = pd.DataFrame(index=raw.index)
    out["div"] = raw["Div"].astype(str).str.strip() if "Div" in raw.columns else div_fallback
    day = pd.to_datetime(raw["Date"].astype(str).str.strip(), dayfirst=True, format="mixed", errors="coerce")
    if "Time" in raw.columns:
        t = pd.to_timedelta(raw["Time"].astype(str).str.strip() + ":00", errors="coerce")
    else:
        t = pd.Series(pd.NaT, index=raw.index, dtype="timedelta64[ns]")
    out["date"] = day + t.fillna(pd.Timedelta(hours=15))
    out["home"] = raw["HomeTeam"].astype(str).str.strip()
    out["away"] = raw["AwayTeam"].astype(str).str.strip()
    out["hg"] = pd.to_numeric(raw["FTHG"], errors="coerce") if "FTHG" in raw.columns else np.nan
    out["ag"] = pd.to_numeric(raw["FTAG"], errors="coerce") if "FTAG" in raw.columns else np.nan

    for c in ("odds_h", "odds_d", "odds_a"):
        out[c] = np.nan
    for cols in ODDS_SETS:
        if not all(c in raw.columns for c in cols):
            continue
        vals = raw[list(cols)].apply(pd.to_numeric, errors="coerce")
        ok = (vals > 1.0).all(axis=1) & out["odds_h"].isna()
        for dst, src in zip(("odds_h", "odds_d", "odds_a"), cols):
            out.loc[ok, dst] = vals.loc[ok, src]

    out = out.dropna(subset=["date"])
    return out[COLUMNS].sort_values("date").reset_index(drop=True)


def fetch_history(leagues, n_seasons: int, today: pd.Timestamp, cache_path: Path) -> pd.DataFrame:
    """Dohrané zápasy za posledních n sezón. Při výpadku zdroje použije dřívější uloženou kopii."""
    frames = []
    for div in leagues:
        for code in season_codes(today, n_seasons):
            url = f"{BASE}/mmz4281/{code}/{div}.csv"
            try:
                content = _download(url)
            except RuntimeError as exc:
                print(f"VAROVÁNÍ: {exc}", file=sys.stderr)
                continue
            if content is None:
                print(f"Soubor {url} neexistuje (přeskakuji).")
                continue
            df = parse_csv(content, div)
            df = df[df["hg"].notna() & df["ag"].notna()]
            df["div"] = div
            frames.append(df)
            print(f"{div} {code}: {len(df)} zápasů")
    parts = []
    if Path(cache_path).exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        parts.append(cached[cached["div"].isin(leagues)])
    parts.extend(frames)
    parts = [p for p in parts if len(p)]
    if not parts:
        raise RuntimeError("Nepodařilo se získat žádná historická data a není ani uložená kopie.")
    hist = pd.concat(parts, ignore_index=True)
    hist = hist.drop_duplicates(["div", "date", "home", "away"], keep="last")
    return hist.sort_values("date").reset_index(drop=True)


def fetch_fixtures(leagues) -> pd.DataFrame:
    """Nadcházející zápasy (fixtures.csv obsahuje hlavní evropské ligy na nejbližší dny)."""
    try:
        content = _download(f"{BASE}/fixtures.csv")
    except RuntimeError as exc:
        print(f"VAROVÁNÍ: {exc}", file=sys.stderr)
        return pd.DataFrame(columns=COLUMNS)
    if content is None:
        return pd.DataFrame(columns=COLUMNS)
    df = parse_csv(content)
    df = df[df["div"].isin(leagues)].copy()
    df["hg"] = np.nan
    df["ag"] = np.nan
    return df.reset_index(drop=True)
