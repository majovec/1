"""Nastavení projektu. Vše lze přepsat proměnnými prostředí (v GitHubu: Settings → Variables)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DATA_DIR = ROOT / "docs" / "data"

# Kódy líg na football-data.co.uk (zdarma, bez API klíče).
LEAGUE_NAMES = {
    "E0": "Premier League (ENG)", "E1": "Championship (ENG)",
    "D1": "Bundesliga (GER)", "D2": "2. Bundesliga (GER)",
    "SP1": "La Liga (ESP)", "SP2": "Segunda División (ESP)",
    "I1": "Serie A (ITA)", "I2": "Serie B (ITA)",
    "F1": "Ligue 1 (FRA)", "F2": "Ligue 2 (FRA)",
    "N1": "Eredivisie (NED)", "B1": "Pro League (BEL)",
    "P1": "Primeira Liga (POR)", "T1": "Süper Lig (TUR)",
    "G1": "Super League (GRE)", "SC0": "Premiership (SCO)",
}


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


LEAGUES = [x.strip().upper() for x in _env("LEAGUES", "E0").split(",") if x.strip()]
N_SEASONS = int(_env("N_SEASONS", "4"))              # kolik sezón historie stáhnout
HALF_LIFE_DAYS = float(_env("HALF_LIFE_DAYS", "450"))  # za jak dlouho klesne váha zápasu na polovinu
MIN_TRAIN_MATCHES = int(_env("MIN_TRAIN_MATCHES", "300"))  # kolik zápasů na ligu se použije jen k trénování
EDGE_THRESHOLD = float(_env("EDGE_THRESHOLD", "0.05"))     # od jaké výhody oproti kurzu se zápas označí
UPCOMING_DAYS = int(_env("UPCOMING_DAYS", "10"))           # na kolik dní dopředu předpovídat
