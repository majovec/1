"""Tenis: síla hráčů (Elo celkové + podle povrchu), backtest proti kurzům, předpovědi nadcházejících zápasů.

Historie + kurzy: tennis-data.co.uk (ATP i WTA, xlsx). Nadcházející zápasy: ESPN (bez kurzů).
Zdroje se mohou změnit; chyba v tenise nikdy neshodí fotbalovou analýzu (pipeline ji zachytí).
"""
import io
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from app import espn
from app.export import CZ_SHIFT, write_json

HEADERS = {"User-Agent": "football-ai-analyst/5 (GitHub Actions)"}
ESPN_TENNIS = "https://site.api.espn.com/apis/site/v2/sports/tennis"
TOURS = {"atp": "{y}/{y}.xlsx", "wta": "{y}w/{y}.xlsx"}
ODDS_SETS = [("AvgW", "AvgL"), ("B365W", "B365L"), ("PSW", "PSL")]
COLS = ["tour", "date", "tournament", "surface", "round", "w", "l", "odds_w", "odds_l"]
K_BASE, K_SHIFT, K_POW, SURF_MIX, START = 250.0, 5.0, 0.4, 0.5, 1500.0
VERSION = "tennis-elo-v1"


# ---------- data ----------
def parse_frame(raw: pd.DataFrame, tour: str) -> pd.DataFrame:
    need = {"Date", "Winner", "Loser"}
    if not need <= set(raw.columns):
        return pd.DataFrame(columns=COLS)
    raw = raw.dropna(subset=list(need)).copy()
    out = pd.DataFrame(index=raw.index)
    out["tour"] = tour
    out["date"] = pd.to_datetime(raw["Date"], errors="coerce")
    out["tournament"] = raw["Tournament"].astype(str).str.strip() if "Tournament" in raw else ""
    out["surface"] = raw["Surface"].astype(str).str.strip() if "Surface" in raw else ""
    out["round"] = raw["Round"].astype(str).str.strip() if "Round" in raw else ""
    out["w"], out["l"] = raw["Winner"].astype(str).str.strip(), raw["Loser"].astype(str).str.strip()
    out["odds_w"] = np.nan
    out["odds_l"] = np.nan
    for cw, cl in ODDS_SETS:
        if cw in raw and cl in raw:
            ow, ol = pd.to_numeric(raw[cw], errors="coerce"), pd.to_numeric(raw[cl], errors="coerce")
            ok = (ow > 1) & (ol > 1) & out["odds_w"].isna()
            out.loc[ok, "odds_w"], out.loc[ok, "odds_l"] = ow[ok], ol[ok]
    # kontrolní: dohrané zápasy bez Comment (odstoupení/skreč se do ratingu nepočítají)
    if "Comment" in raw:
        out = out[raw["Comment"].astype(str).str.lower().isin(["completed", "nan", ""])]
    return out.dropna(subset=["date"])[COLS]


def _download(url: str):
    for u in (url, url.replace("https://", "http://")):
        try:
            r = requests.get(u, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                return r.content
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return None


def fetch_history(today: pd.Timestamp, years: int = 4, cache: Path = None) -> pd.DataFrame:
    frames = []
    for tour, pat in TOURS.items():
        for y in range(today.year, today.year - years, -1):
            content = _download("https://www.tennis-data.co.uk/" + pat.format(y=y))
            if content is None:
                print(f"Tenis: {tour} {y} nelze stáhnout (přeskakuji).", file=sys.stderr)
                continue
            try:
                df = parse_frame(pd.read_excel(io.BytesIO(content)), tour)
            except Exception as exc:  # noqa: BLE001
                print(f"Tenis: {tour} {y} nelze přečíst: {exc}", file=sys.stderr)
                continue
            frames.append(df)
            print(f"Tenis {tour} {y}: {len(df)} zápasů")
    if cache is not None and Path(cache).exists():
        frames.append(pd.read_csv(cache, parse_dates=["date"]))
    frames = [f for f in frames if len(f)]
    if not frames:
        raise RuntimeError("Tenis: žádná historická data.")
    h = pd.concat(frames, ignore_index=True).drop_duplicates(["tour", "date", "w", "l"], keep="first")
    if cache is not None:
        h.to_csv(cache, index=False)
    return h.sort_values("date").reset_index(drop=True)


# ---------- Elo ----------
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class Elo:
    def __init__(self):
        self.r, self.n = defaultdict(lambda: START), defaultdict(int)
        self.rs, self.ns = defaultdict(lambda: START), defaultdict(int)

    def p_first(self, a, b, surf):
        d = (1 - SURF_MIX) * (self.r[a] - self.r[b])
        if surf:
            d += SURF_MIX * (self.rs[(a, surf)] - self.rs[(b, surf)])
        else:
            d += SURF_MIX * (self.r[a] - self.r[b])
        return float(_sig(d * np.log(10) / 400))

    def update(self, w, l, surf):
        pa = float(_sig((self.r[w] - self.r[l]) * np.log(10) / 400))
        kw, kl = K_BASE / (self.n[w] + K_SHIFT) ** K_POW, K_BASE / (self.n[l] + K_SHIFT) ** K_POW
        self.r[w] += kw * (1 - pa); self.r[l] -= kl * (1 - pa)
        self.n[w] += 1; self.n[l] += 1
        if surf:
            ps = float(_sig((self.rs[(w, surf)] - self.rs[(l, surf)]) * np.log(10) / 400))
            ksw, ksl = K_BASE / (self.ns[(w, surf)] + K_SHIFT) ** K_POW, K_BASE / (self.ns[(l, surf)] + K_SHIFT) ** K_POW
            self.rs[(w, surf)] += ksw * (1 - ps); self.rs[(l, surf)] -= ksl * (1 - ps)
            self.ns[(w, surf)] += 1; self.ns[(l, surf)] += 1


def walk(hist: pd.DataFrame):
    """Chronologicky projde zápasy; vrací (tabulka s předzápasovou pravděpodobností vítěze, {tour: Elo})."""
    elos, rows = {}, []
    for r in hist.sort_values("date").itertuples():
        e = elos.setdefault(r.tour, Elo())
        surf = r.surface if r.surface and r.surface != "nan" else ""
        rows.append((r.tour, r.date, r.w, r.l, e.p_first(r.w, r.l, surf), e.n[r.w], e.n[r.l], r.odds_w, r.odds_l))
        e.update(r.w, r.l, surf)
    cols = ["tour", "date", "w", "l", "p_w", "n_w", "n_l", "odds_w", "odds_l"]
    return pd.DataFrame(rows, columns=cols), elos


def _ll(p):
    return float(-np.log(np.clip(p, 1e-9, 1)).mean())


def backtest(tbl: pd.DataFrame, today: pd.Timestamp, test_years: float = 2.0, min_games: int = 8) -> dict:
    t = tbl[(tbl["date"] >= today - pd.Timedelta(days=int(365 * test_years))) &
            (tbl["n_w"] >= min_games) & (tbl["n_l"] >= min_games)].copy()
    out = {"matches": int(len(t)), "weight": 0.0}
    if len(t) < 200:
        out["note"] = "málo zápasů na spolehlivé otestování"
        return out
    pw = t["p_w"].to_numpy()
    out.update(accuracy=float((pw > 0.5).mean()), log_loss=_ll(pw), naive_log_loss=_ll(np.full(len(pw), 0.5)))
    has = t[["odds_w", "odds_l"]].notna().all(axis=1).to_numpy()
    if has.sum() >= 200:
        ow, ol = t.loc[has, "odds_w"].to_numpy(float), t.loc[has, "odds_l"].to_numpy(float)
        mk = (1 / ow) / (1 / ow + 1 / ol)
        pm = pw[has]
        cut = int(len(mk) * 0.7)                       # váhu trhu se učí na starších, ověří na novějších zápasech
        grid = np.linspace(0, 1, 11)
        best = grid[int(np.argmin([_ll(g * mk[:cut] + (1 - g) * pm[:cut]) for g in grid]))]
        out.update(market_n=int(has.sum()), market_log_loss=_ll(mk), market_accuracy=float((mk > 0.5).mean()),
                   model_log_loss_on_market=_ll(pm), model_accuracy_on_market=float((pm > 0.5).mean()),
                   weight=float(best), blend_log_loss_test=_ll(best * mk[cut:] + (1 - best) * pm[cut:]),
                   model_log_loss_test=_ll(pm[cut:]), market_log_loss_test=_ll(mk[cut:]))
    cal = []
    edges = np.linspace(0.5, 1, 6)
    pmax = np.maximum(pw, 1 - pw)
    win = np.where(pw >= 0.5, 1.0, 0.0)               # vyšlo to, když favorit (dle modelu) vyhrál
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (pmax >= lo) & (pmax < hi + (1e-9 if hi == 1 else 0))
        if s.sum() >= 20:
            cal.append({"predicted": float(pmax[s].mean()), "observed": float(win[s].mean()), "n": int(s.sum())})
    out["calibration"] = cal
    return out


# ---------- jména a ESPN ----------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def build_index(hist: pd.DataFrame) -> dict:
    """'Sinner J.' → klíč ('sinner', 'j'). Hodnota = jméno tak, jak je v historii."""
    idx = {}
    for name in pd.unique(pd.concat([hist["w"], hist["l"]])):
        m = re.match(r"^(.*?)\s+([A-Z]\.(?:[A-Z]\.)*)$", str(name).strip())
        if m:
            idx[(_norm(m.group(1)), m.group(2)[0].lower())] = name
    return idx


def match_name(espn_name: str, idx: dict):
    toks = str(espn_name).replace("-", "-").split()
    if len(toks) < 2:
        return None
    ini = _norm(toks[0])[:1]
    for k in (1, 2, 3):
        if len(toks) > k:
            hit = idx.get((_norm("".join(toks[-k:])), ini))
            if hit:
                return hit
    return None


def parse_espn_tennis(js: dict, tour: str) -> list:
    out = []
    for ev in (js or {}).get("events", []) or []:
        groups = ev.get("groupings") or [{"grouping": {"displayName": "Singles"}, "competitions": ev.get("competitions", [])}]
        for g in groups:
            gname = str((g.get("grouping") or {}).get("displayName", "")).lower()
            if "double" in gname or "mixed" in gname or "doubles" in gname:
                continue
            for c in g.get("competitions", []) or []:
                st = (c.get("status") or {}).get("type", {})
                if st.get("state") != "pre":
                    continue
                names = [((x.get("athlete") or {}).get("displayName") or (x.get("team") or {}).get("displayName"))
                         for x in c.get("competitors", [])]
                if len(names) != 2 or not all(names):
                    continue
                try:
                    when = pd.Timestamp(c.get("date") or c.get("startDate"))
                    when = (when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC"))
                    when = when.tz_convert("Europe/London").tz_localize(None)
                except Exception:  # noqa: BLE001
                    continue
                out.append({"tour": tour, "date": when, "tournament": ev.get("name", ""), "a": names[0], "b": names[1]})
    return out


def fetch_upcoming(now: pd.Timestamp, days: int = 3) -> list:
    rows = []
    for tour in TOURS:
        url = f"{ESPN_TENNIS}/{tour}/scoreboard"
        try:
            st, js = espn._request(url, {"dates": f"{now:%Y%m%d}-{now + pd.Timedelta(days=days):%Y%m%d}"})
        except RuntimeError as exc:
            print(f"VAROVÁNÍ: tenis ESPN {tour}: {exc}", file=sys.stderr)
            continue
        if st != 200:
            print(f"VAROVÁNÍ: tenis ESPN {tour} vrátilo HTTP {st}", file=sys.stderr)
            continue
        got = parse_espn_tennis(js, tour)
        print(f"Tenis ESPN {tour}: {len(got)} nadcházejících zápasů")
        rows += got
    return rows


def surface_for(tournament: str, surf_map: dict) -> str:
    t = _norm(tournament)
    for k, v in surf_map.items():
        if k and (k in t or t in k) and len(t) > 3:
            return v
    return ""


def predict_upcoming(raw: list, hist: pd.DataFrame, elos: dict, now: pd.Timestamp, days: int) -> list:
    idx = build_index(hist)
    surf_map = {_norm(t): s for t, s in hist.dropna(subset=["surface"]).drop_duplicates("tournament", keep="last")[["tournament", "surface"]].itertuples(index=False)}
    rows, skipped = [], 0
    for m in raw:
        if not (now < m["date"] <= now + pd.Timedelta(days=days)):
            continue
        a, b = match_name(m["a"], idx), match_name(m["b"], idx)
        e = elos.get(m["tour"])
        if not a or not b or e is None:
            skipped += 1
            continue
        surf = surface_for(m["tournament"], surf_map)
        p = e.p_first(a, b, surf if surf != "nan" else "")
        rows.append({"tour": m["tour"], "date": m["date"], "tournament": m["tournament"], "surface": surf,
                     "a": a, "b": b, "a_full": m["a"], "b_full": m["b"], "p_a": p,
                     "games": [int(e.n[a]), int(e.n[b])], "rating": [round(e.r[a]), round(e.r[b])]})
    print(f"Tenis: předpovězeno {len(rows)}, přeskočeno (neznámý hráč): {skipped}")
    return sorted(rows, key=lambda r: r["date"])


# ---------- vlastní záznam předpovědí ----------
LOG_COLS = ["key", "date", "tour", "a", "b", "p_a", "surface", "created_at"]


def update_log(path: Path, preds: list, now: pd.Timestamp) -> pd.DataFrame:
    log = pd.read_csv(path, parse_dates=["date", "created_at"]) if Path(path).exists() else pd.DataFrame(columns=LOG_COLS)
    have = {k: i for i, k in enumerate(log["key"])}
    new = []
    for r in preds:
        key = f"{r['tour']}|{r['date']:%Y-%m-%d}|{'|'.join(sorted([r['a'], r['b']]))}"
        if key in have:
            i = have[key]
            if log.at[i, "date"] > now:               # odehrané se už nepřepisují
                log.loc[i, ["p_a", "surface"]] = [r["p_a"], r["surface"]]
            continue
        new.append({"key": key, "date": r["date"], "tour": r["tour"], "a": r["a"], "b": r["b"],
                    "p_a": r["p_a"], "surface": r["surface"], "created_at": now})
    if new:
        log = pd.concat([log, pd.DataFrame(new)], ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    log[LOG_COLS].to_csv(path, index=False)
    return log


def settle(log: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    if log.empty:
        return pd.DataFrame()
    rows = []
    for r in log.itertuples():
        c = hist[(hist["tour"] == r.tour) & (hist["date"] >= r.date.normalize() - pd.Timedelta(days=1)) &
                 (hist["date"] <= r.date.normalize() + pd.Timedelta(days=4)) &
                 (((hist["w"] == r.a) & (hist["l"] == r.b)) | ((hist["w"] == r.b) & (hist["l"] == r.a)))]
        if len(c):
            a_won = c.iloc[0]["w"] == r.a
            p_win = r.p_a if a_won else 1 - r.p_a
            rows.append({"date": r.date, "tour": r.tour, "a": r.a, "b": r.b, "p_a": r.p_a, "winner": c.iloc[0]["w"],
                         "correct": bool((r.p_a > 0.5) == a_won), "p_winner": p_win})
    return pd.DataFrame(rows)


# ---------- celý běh ----------
def run(now: pd.Timestamp, data_dir: Path, docs_dir: Path, upcoming_days: int = 3, hist: pd.DataFrame = None,
        raw_upcoming: list = None, log=print) -> dict:
    data_dir, docs_dir = Path(data_dir), Path(docs_dir)
    if hist is None:
        hist = fetch_history(now, cache=data_dir / "tennis_matches.csv")
    tbl, elos = walk(hist)
    bt = backtest(tbl, now)
    log(f"Tenis backtest: {bt.get('matches')} zápasů, přesnost {bt.get('accuracy')}")
    raw = fetch_upcoming(now, upcoming_days) if raw_upcoming is None else raw_upcoming
    preds = predict_upcoming(raw, hist, elos, now, upcoming_days)
    tlog = update_log(data_dir / "tennis_log.csv", preds, now)
    settled = settle(tlog, hist)
    live = {"settled": int(len(settled)), "pending": int((tlog["date"] > now).sum()) if len(tlog) else 0}
    if len(settled):
        live.update(accuracy=float(settled["correct"].mean()), log_loss=_ll(settled["p_winner"].to_numpy()))
    for r in preds:
        r["date"] = r["date"] + CZ_SHIFT
    srows = [] if settled.empty else [dict(r, date=r["date"] + CZ_SHIFT) for r in settled.sort_values("date", ascending=False).head(100).to_dict("records")]
    summary = {"generated_at": now, "version": VERSION, "history_matches": int(len(hist)), "backtest": bt, "live": live,
               "history_to": hist["date"].max()}
    write_json(docs_dir / "tennis.json", {"summary": summary, "upcoming": preds, "settled": srows})
    return summary
