# Football AI Analyst v3

Fotbalový analytik, který běží **zdarma jen na GitHubu** (Actions + Pages), bez serveru a bez API klíče.

## Co dělá
1. **Učí se z historie:** stáhne posledních 4 sezóny z football-data.co.uk (výsledky + průměrné kurzy bookmakerů).
2. **Ověřuje sám sebe (backtest):** projde minulé zápasy po dnech a každý předpoví jen z toho, co bylo před ním. Porovná se s naivním odhadem a s kurzy bookmakerů.
3. **Předpovídá nadcházející zápasy:** 1/X/2, očekávané góly, Over 2,5, BTTS, nejpravděpodobnější skóre a rozdíl proti kurzu.
4. **Sleduje vlastní úspěšnost:** každou předpověď uloží do `data/predictions_log.csv`. Po odehrání zápasu ji sám vyhodnotí. Odehrané predikce se už nepřepisují, takže se výsledky nedají zpětně vylepšit.
5. **Dashboard na GitHub Pages** (`docs/index.html`) s tlačítkem **▶ Analyzovat aktuální zápasy**.

## Zapnutí (jednorázově)
1. Repozitář musí být **Public** (GitHub Pages zdarma funguje jen na veřejném repozitáři; v něm nejsou žádná tajemství).
2. Nahraj ZIP do repozitáře a rozbal ho přes workflow *Unpack ZIP* (soubory jsou v kořeni ZIPu).
3. **Workflow soubory:** GitHub nedovolí botovi vytvářet `.github/workflows`, proto jsou ve složce `_workflows/`. Otevři `_workflows/analyze.yml` → ✏️ (Edit) → v názvu souboru nahoře změň cestu na `.github/workflows/analyze.yml` → Commit. Stejně můžeš udělat s `tests.yml` (nepovinné).
4. **Settings → Pages:** Source = *Deploy from a branch*, Branch = `main`, složka `/docs`.
5. **Actions → Analyze → Run workflow.** Po 2–3 minutách se data uloží do repozitáře a dashboard na `https://<uživatel>.github.io/<repozitář>/` se naplní.
6. Dál se analýza spouští sama 2× denně (06:20 a 18:20 UTC) a kdykoli ručně.

## Tlačítko na dashboardu
- Bez nastavení otevře stránku Actions, kde klikneš *Run workflow*.
- **Na jedno kliknutí:** vytvoř *fine-grained token* (GitHub → Settings → Developer settings → Fine-grained tokens), jen pro tento repozitář, oprávnění **Actions: Read and write**. Vlož ho tlačítkem *Token* na dashboardu. Uloží se jen v tvém prohlížeči. Token drž jen pro tento repozitář a nikomu ho neukazuj.

## Ligy
Výchozí je Premier League (`E0`). Další: `D1` Bundesliga, `SP1` La Liga, `I1` Serie A, `F1` Ligue 1, `N1`, `P1`, `B1`, `T1` … (seznam v `config.py`).
Nastavení: **Settings → Secrets and variables → Actions → Variables → New** → `LEAGUES` = `E0,D1,SP1,I1,F1`.
Další volitelné proměnné: `N_SEASONS`, `HALF_LIFE_DAYS`, `EDGE_THRESHOLD`, `UPCOMING_DAYS`.

## Jak číst výsledky
- **Log loss** (nižší = lepší) je hlavní měřítko. Porovnávej model s „naivně“ a s „kurzy“. Pokud kurzy vyhrávají (běžné), model sám o sobě peníze nevydělá.
- **Kalibrace:** body mají ležet na úhlopříčce.
- **Sázky na hodnotu** jsou jen simulace nad historií. Opravdová zkouška je sekce *Úspěšnost mých vlastních předpovědí*; potřebuje stovky zápasů, tedy týdny až měsíce.

## Lokálně
```bash
pip install -r requirements.txt pytest
pytest -q
python -m app.pipeline --demo   # syntetická data, funguje offline
python -m app.pipeline          # skutečná data
```

## Limity
- Model je jednoduchý: góly, síla týmů s časovým útlumem, domácí prostředí a Dixon–Coles korekce. Nezná zranění, sestavy, xG ani únavu.
- Zdroj zveřejňuje jen zápasy na pár dní dopředu. Čas zápasů je britský.
- Nic z toho není finanční poradenství a výsledky nejsou garance.
