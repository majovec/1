"""Nastavení projektu. Vše lze přepsat proměnnými prostředí (v GitHubu: Settings → Variables)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DATA_DIR = ROOT / "docs" / "data"

# Kódy líg: (země, vlajka, název). Kódy E0…SC0 jsou z football-data.co.uk (mají kurzy bookmakerů),
# kódy z ESPN_LEAGUES se berou z veřejného ESPN API (bez kurzů).
LEAGUE_INFO = {
    "CZ1": ("Česko", "🇨🇿", "Fortuna liga"),
    "E0": ("Anglie", "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Premier League"), "E1": ("Anglie", "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Championship"),
    "D1": ("Německo", "🇩🇪", "Bundesliga"), "D2": ("Německo", "🇩🇪", "2. Bundesliga"),
    "SP1": ("Španělsko", "🇪🇸", "La Liga"), "SP2": ("Španělsko", "🇪🇸", "Segunda División"),
    "I1": ("Itálie", "🇮🇹", "Serie A"), "I2": ("Itálie", "🇮🇹", "Serie B"),
    "F1": ("Francie", "🇫🇷", "Ligue 1"), "F2": ("Francie", "🇫🇷", "Ligue 2"),
    "N1": ("Nizozemsko", "🇳🇱", "Eredivisie"), "B1": ("Belgie", "🇧🇪", "Pro League"),
    "P1": ("Portugalsko", "🇵🇹", "Primeira Liga"), "T1": ("Turecko", "🇹🇷", "Süper Lig"),
    "G1": ("Řecko", "🇬🇷", "Super League"), "SC0": ("Skotsko", "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Premiership"),
    "AT1": ("Rakousko", "🇦🇹", "Bundesliga"), "DK1": ("Dánsko", "🇩🇰", "Superliga"),
    "PL1": ("Polsko", "🇵🇱", "Ekstraklasa"), "SE1": ("Švédsko", "🇸🇪", "Allsvenskan"),
    "NO1": ("Norsko", "🇳🇴", "Eliteserien"), "CH1": ("Švýcarsko", "🇨🇭", "Super League"),
    "SK1": ("Slovensko", "🇸🇰", "Niké liga"),
}
LEAGUE_NAMES = {k: f"{v[2]} ({v[0]})" for k, v in LEAGUE_INFO.items()}

# Ligy bez football-data.co.uk: kód → ESPN slug (https://site.api.espn.com/apis/site/v2/sports/soccer/<slug>/scoreboard)
ESPN_LEAGUES = {"AT1": "aut.1", "DK1": "den.1", "SE1": "swe.1", "NO1": "nor.1"}
# Ligy z TheSportsDB (bezplatné API): kód → (země pro vyhledání, klíčová slova v názvu ligy). ESPN české, polské ani švýcarské ligy nevrací.
TSDB_LEAGUES = {"CZ1": ("Czech_Republic", ["Czech First", "Fortuna", "Chance"]),
                "PL1": ("Poland", ["Ekstraklasa", "Polish"]),
                "CH1": ("Switzerland", ["Swiss Super", "Super League"]),
                "SK1": ("Slovakia", ["Slovak First", "Niké", "Nike", "Fortuna"])}

DEFAULT_LEAGUES = "CZ1,E0,D1,SP1,I1,F1,N1,B1,P1,T1,SC0,AT1,DK1"


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


LEAGUES = [x.strip().upper() for x in _env("LEAGUES", DEFAULT_LEAGUES).split(",") if x.strip()]
N_SEASONS = int(_env("N_SEASONS", "4"))              # kolik sezón historie stáhnout
HALF_LIFE_DAYS = float(_env("HALF_LIFE_DAYS", "450"))  # za jak dlouho klesne váha zápasu na polovinu
MIN_TRAIN_MATCHES = int(_env("MIN_TRAIN_MATCHES", "300"))  # kolik zápasů na ligu se použije jen k trénování
EDGE_THRESHOLD = float(_env("EDGE_THRESHOLD", "0.05"))     # od jaké výhody oproti kurzu se zápas označí
UPCOMING_DAYS = int(_env("UPCOMING_DAYS", "10"))           # na kolik dní dopředu předpovídat
MARKET_WEIGHT = float(_env("MARKET_WEIGHT", "0.7"))        # jak moc věřit kurzům trhu při míchání s modelem (0–1)
