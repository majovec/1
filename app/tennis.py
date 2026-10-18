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


BROWSER = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}


def _download(urls):
    """Zkusí postupně všechny adresy; vypíše, proč která selhala."""
    for u in ([urls] if isinstance(urls, str) else urls):
        try:
            r = requests.get(u, headers=BROWSER, timeout=60)
            if r.status_code == 200 and len(r.content) > 500:
                return r.content
            print(f"  {u} → HTTP {r.status_code}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"  {u} → {type(exc).__name__}: {str(exc)[:80]}", file=sys.stderr)
    return None


ROUND_DAY = {"R128": 0, "R64": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5, "F": 6, "RR": 2, "BR": 5}
SACK = {"atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{y}.csv",
        "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{y}.csv"}


def parse_sackmann(raw: pd.DataFrame, tour: str) -> pd.DataFrame:
    """Jeff Sackmann (GitHub, CC BY-NC-SA): plná jména, bez kurzů. Datum = začátek turnaje + posun podle kola."""
    need = {"tourney_date", "winner_name", "loser_name"}
    if not need <= set(raw.columns):
        return pd.DataFrame(columns=COLS)
    raw = raw.dropna(subset=list(need)).copy()
    score = raw["score"].astype(str).str.upper() if "score" in raw else pd.Series("", index=raw.index)
    raw = raw[~score.str.contains("RET|W/O|DEF|WALKOVER|ABD|UNP", regex=True)]
    rnd = raw["round"].astype(str) if "round" in raw else pd.Series("", index=raw.index)
    out = pd.DataFrame(index=raw.index)
    out["tour"] = tour
    out["date"] = pd.to_datetime(raw["tourney_date"].astype(int).astype(str), format="%Y%m%d", errors="coerce") + \
        pd.to_timedelta(rnd.map(ROUND_DAY).fillna(0).astype(int), unit="D")
    out["tournament"] = raw["tourney_name"].astype(str) if "tourney_name" in raw else ""
    out["surface"] = raw["surface"].astype(str) if "surface" in raw else ""
    out["round"] = rnd
    out["w"], out["l"] = raw["winner_name"].astype(str).str.strip(), raw["loser_name"].astype(str).str.strip()
    out["odds_w"], out["odds_l"] = np.nan, np.nan
    return out.dropna(subset=["date"])[COLS]


_SACK_FILES = {}


def sack_files(repo: str, prefix: str) -> dict:
    """Seznam souborů {rok: download_url} přímo z GitHub API (nemusíme hádat větev ani cestu)."""
    if repo in _SACK_FILES:
        return _SACK_FILES[repo]
    import os
    hdr = dict(BROWSER, Accept="application/vnd.github+json")
    if os.environ.get("GITHUB_TOKEN"):
        hdr["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    out = {}
    try:
        r = requests.get(f"https://api.github.com/repos/JeffSackmann/{repo}/contents/", headers=hdr, timeout=60)
        if r.status_code == 200:
            for it in r.json():
                m = re.fullmatch(rf"{prefix}_matches_(\d{{4}})\.csv", it.get("name", ""))
                if m and it.get("download_url"):
                    out[int(m.group(1))] = it["download_url"]
            print(f"Tenis (sack) {repo}: nalezeno souborů: {len(out)}" + (f", nejnovější rok {max(out)}" if out else ""))
        else:
            print(f"  GitHub API {repo} → HTTP {r.status_code} {r.text[:120]}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"  GitHub API {repo}: {type(exc).__name__}: {str(exc)[:80]}", file=sys.stderr)
    _SACK_FILES[repo] = out
    return out


def _collect(today, years, source):
    frames = []
    for tour in TOURS:
        for y in range(today.year, today.year - years, -1):
            if source == "td":
                urls = [f"{pre}://{host}/{TOURS[tour].format(y=y)}" for pre in ("https", "http") for host in ("www.tennis-data.co.uk", "tennis-data.co.uk")]
            else:
                found = sack_files(f"tennis_{tour}", tour).get(y)
                urls = ([found] if found else []) + [SACK[tour].format(y=y), SACK[tour].format(y=y).replace("/master/", "/refs/heads/master/"),
                                                      SACK[tour].format(y=y).replace("/master/", "/main/")]
            content = _download(urls)
            if content is None:
                print(f"Tenis ({source}): {tour} {y} nelze stáhnout.", file=sys.stderr)
                continue
            try:
                raw = pd.read_excel(io.BytesIO(content)) if source == "td" else pd.read_csv(io.BytesIO(content), low_memory=False)
                df = parse_frame(raw, tour) if source == "td" else parse_sackmann(raw, tour)
            except Exception as exc:  # noqa: BLE001
                print(f"Tenis ({source}): {tour} {y} nelze přečíst: {exc}", file=sys.stderr)
                continue
            frames.append(df)
            print(f"Tenis ({source}) {tour} {y}: {len(df)} zápasů")
    return [f for f in frames if len(f)]


CLAY = ("roland garros", "french open", "monte", "rome", "italian open", "madrid", "barcelona", "hamburg", "bastad", "gstaad", "kitzbuhel",
        "umag", "estoril", "marrakech", "houston", "buenos aires", "rio ", "santiago", "bucharest", "geneva", "lyon", "parma", "palermo",
        "bogota", "cordoba", "munich", "belgrade", "sardinia", "prague", "rabat", "charleston", "strasbourg", "lausanne", "bad gastein",
        "warsaw", "budapest", "iasi", "cluj", "portoroz", "montreal 2", "clay")
GRASS = ("wimbledon", "halle", "queen", "hertogenbosch", "stuttgart", "mallorca", "eastbourne", "newport", "birmingham", "nottingham",
         "berlin", "bad homburg", "grass", "libema", "surbiton", "ilkley")


def guess_surface(tournament: str) -> str:
    """ESPN povrch neuvádí; odhad podle názvu turnaje (kolísá u výjimek, váha povrchu v modelu je proto jen poloviční)."""
    n = str(tournament).lower()
    if any(k in n for k in GRASS):
        return "Grass"
    if any(k in n for k in CLAY):
        return "Clay"
    return "Hard"


_DIAG = {"done": False}


def parse_results(js: dict, tour: str) -> pd.DataFrame:
    """Dohrané dvouhry z ESPN scoreboardu (bez skečů a walkoverů)."""
    out = []
    for ev in (js or {}).get("events", []) or []:
        groups = ev.get("groupings") or [{"grouping": {"displayName": "Singles"}, "competitions": ev.get("competitions", [])}]
        for g in groups:
            gname = str((g.get("grouping") or {}).get("displayName", "")).lower()
            if "double" in gname or "mixed" in gname:
                continue
            for c in g.get("competitions", []) or []:
                st = (c.get("status") or {}).get("type", {})
                if not (st.get("completed") or st.get("state") == "post"):
                    continue
                detail = " ".join(str(st.get(k, "")) for k in ("detail", "description", "shortDetail")).lower()
                if any(k in detail for k in ("retire", "walkover", "w/o", "default", "cancel", "postpon", "suspend", "abandon")):
                    continue
                comps = c.get("competitors") or []
                if len(comps) != 2:
                    continue
                names = [((x.get("athlete") or {}).get("displayName") or "").strip() for x in comps]
                wins = [bool(x.get("winner")) for x in comps]
                if sum(wins) != 1 or not all(names):
                    continue
                try:
                    when = pd.Timestamp(c.get("date") or c.get("startDate") or ev.get("date"))
                    when = when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
                    when = when.tz_convert("Europe/London").tz_localize(None)
                except Exception:  # noqa: BLE001
                    continue
                rd = c.get("round")
                rd = rd.get("displayName", "") if isinstance(rd, dict) else str(rd or "")
                w, l = (names[0], names[1]) if wins[0] else (names[1], names[0])
                tn = ev.get("name", "")
                out.append({"tour": tour, "date": when, "tournament": tn, "surface": guess_surface(tn), "round": rd,
                            "w": w, "l": l, "odds_w": np.nan, "odds_l": np.nan})
    return pd.DataFrame(out, columns=COLS)


def espn_history(today: pd.Timestamp, days: int, cache: Path = None) -> pd.DataFrame:
    """Výsledky ATP + WTA po dnech z ESPN (paralelně). Při dalších bězích se stahuje jen od posledního uloženého dne."""
    from concurrent.futures import ThreadPoolExecutor
    old = pd.read_csv(cache, parse_dates=["date"]) if cache is not None and Path(cache).exists() else None
    start = today.normalize() - pd.Timedelta(days=days)
    if old is not None and len(old):
        start = max(start, old["date"].max().normalize() - pd.Timedelta(days=4))
    jobs = [(tour, d) for tour in TOURS for d in pd.date_range(start, today.normalize())]

    def one(job):
        tour, d = job
        try:
            st, js = espn._request(f"{ESPN_TENNIS}/{tour}/scoreboard", {"dates": f"{d:%Y%m%d}"}, tries=2)
        except RuntimeError:
            return tour, d, 0, None
        return tour, d, st, js

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(one, jobs))
    bad = sum(1 for _, _, st, _ in res if st != 200)
    frames = [parse_results(js, tour) for tour, _, st, js in res if st == 200 and js]
    frames = [f for f in frames if len(f)]
    got = sum(len(f) for f in frames)
    print(f"Tenis (ESPN): {len(jobs)} dotazů, neúspěšných {bad}, nových dohraných zápasů {got}")
    if got == 0 and not _DIAG["done"]:
        _DIAG["done"] = True
        sample = next((js for _, _, st, js in res if st == 200 and js and js.get("events")), None)
        if sample:
            ev = sample["events"][0]
            print(f"  DIAGNOSTIKA: klíče události {list(ev.keys())[:12]}", file=sys.stderr)
            gr = (ev.get("groupings") or [{}])[0]
            print(f"  DIAGNOSTIKA: klíče skupiny {list(gr.keys())[:12]}; první zápas {str((gr.get('competitions') or [{}])[0])[:300]}", file=sys.stderr)
        else:
            print("  DIAGNOSTIKA: žádný den nevrátil žádnou událost", file=sys.stderr)
    frames = ([old] if old is not None else []) + frames
    if not frames:
        raise RuntimeError("Tenis: ESPN nevrátilo žádné výsledky.")
    h = pd.concat(frames, ignore_index=True).drop_duplicates(["tour", "date", "w", "l"], keep="last")
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        h.to_csv(cache, index=False)
    return h.sort_values("date").reset_index(drop=True)


def fetch_history(today: pd.Timestamp, years: int = 4, cache: Path = None) -> pd.DataFrame:
    """1) tennis-data.co.uk (s kurzy), pokud není blokované (z GitHubu vrací HTTP 403); 2) ESPN (bez kurzů).
    Sackmannova data z GitHubu už nejsou dostupná (repozitář vrací 404)."""
    if _download([f"https://www.tennis-data.co.uk/{today.year - 1}/{today.year - 1}.xlsx"]) is not None:
        frames = _collect(today, years, "td")
        if frames:
            h = pd.concat(frames, ignore_index=True).drop_duplicates(["tour", "date", "w", "l"], keep="first")
            return h.sort_values("date").reset_index(drop=True)
    print("Tenis: tennis-data.co.uk nedostupné (kurzy nebudou), používám výsledky z ESPN.")
    esp = None if cache is None else Path(cache).with_name("tennis_espn.csv")
    return espn_history(today, 540, esp)


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
        idx[(_norm(str(name)), "")] = name                      # plné jméno (zdroj Sackmann)
        m = re.match(r"^(.*?)\s+([A-Z]\.(?:[A-Z]\.)*)$", str(name).strip())
        if m:
            idx[(_norm(m.group(1)), m.group(2)[0].lower())] = name
    return idx


def match_name(espn_name: str, idx: dict):
    full = idx.get((_norm(str(espn_name)), ""))
    if full:
        return full
    toks = str(espn_name).split()
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
        got = []
        for d in range(days + 1):                                # po jednom dni: formát, který ESPN přijímá vždy
            day = now.normalize() + pd.Timedelta(days=d)
            try:
                st, js = espn._request(url, {"dates": f"{day:%Y%m%d}"})
            except RuntimeError as exc:
                print(f"VAROVÁNÍ: tenis ESPN {tour}: {exc}", file=sys.stderr)
                continue
            if st != 200:
                print(f"VAROVÁNÍ: tenis ESPN {tour} {day:%Y%m%d} vrátilo HTTP {st}", file=sys.stderr)
                continue
            got += parse_espn_tennis(js, tour)
        got = list({(g["date"], g["a"], g["b"]): g for g in got}.values())
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
