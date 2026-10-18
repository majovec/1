# Football AI Analyst v4

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

## Tenis a automatické obnovení (v6)
- **Tenis** (ATP + WTA): historie výsledků a Elo model, backtest a nadcházející zápasy z ESPN. Externí kurzy pro nadcházející zápasy se načítají samostatně z bezplatného veřejného feedu; nejsou dostupné pro každý zápas a neuvádějí konkrétního bookmakera. Zastaralé nebo nejednoznačně spárované kurzy se nezobrazují. Chyba v tenise nikdy neshodí fotbal.
- **Automaticky:** workflow *Analyze* běží každé 3 hodiny (v `.github/workflows/analyze.yml`; zkopíruj upravený soubor podle kroku 3 výše), dashboard si nová data sám načte každých 5 minut. GitHub může plánované běhy vypnout, když je repozitář dlouho bez aktivity; pak stačí workflow jednou spustit ručně.

## Ligy a země (v4)
Výchozí sada: 🇨🇿 Fortuna liga (`CZ1`), Premier League, Bundesliga, La Liga, Serie A, Ligue 1, Eredivisie, Belgie, Portugalsko, Turecko, Skotsko, Rakousko, Dánsko, Polsko, Švýcarsko.
- **Hlavní ligy** (`E0 D1 SP1 I1 F1 N1 B1 P1 T1 G1 SC0 E1 D2 SP2 I2 F2`) jdou z football-data.co.uk **s kurzy bookmakerů**.
- **Česká liga a menší ligy** (`CZ1 AT1 DK1 PL1 SE1 NO1 CH1 SK1`) jdou z veřejného ESPN API **bez kurzů** (`ESPN_LEAGUES` v `config.py`). Pokud ESPN danou ligu nevrátí, workflow ji přeskočí a vypíše varování do logu.
- Změna sady: **Settings → Secrets and variables → Actions → Variables** → `LEAGUES` = `CZ1,E0,D1`.
- Další proměnné: `N_SEASONS`, `HALF_LIFE_DAYS`, `EDGE_THRESHOLD`, `UPCOMING_DAYS`, `MARKET_WEIGHT` (0–1, výchozí 0,7).

## Co se systém učí (v5)
- **Forma** (body z posledních 5 zápasů) a **odpočinek** (dny od posledního zápasu) se počítají jen z dřívějších zápasů.
- Malá učící vrstva (`app/stack.py`) se při každém běhu znovu naučí z backtestu, jak vážit model, formu, odpočinek a kurzy trhu. Učení se **zapne jen tehdy, když zlepší výsledek na posledních 30 % zápasů, které při učení neviděla**. Výsledek je v `data/stacker.json` a na dashboardu v sekci „Co se systém naučil“.
- Vlastní předpovědi se dál ukládají a vyhodnocují (`data/predictions_log.csv`), nejdou zpětně přepsat.

## Jak vzniká odhad
U lig s kurzy se odhad modelu míchá s kurzy trhu bez marže (70 % trh, 30 % model), protože trh je v praxi nejpřesnější. U lig bez kurzů (Česko) je to jen model, proto je tam spolehlivost označena jako „střední“. Dashboard u každého zápasu ukazuje i **férový kurz** (1 ÷ pravděpodobnost) k porovnání s kurzem, který ti někdo nabízí.

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

## Tenis – externí kurzy

Tenisová část může načítat externí tržní kurzy z veřejného feedu `matches.json` v repozitáři [Mriganka-codes/tennis_data](https://github.com/Mriganka-codes/tennis_data). Tento feed podle svého popisu shromažďuje dnešní zápasy a kurzy z Tennis Exploreru a aktualizuje se přibližně každých šest hodin. **Neuvádí konkrétního bookmakera**, proto je aplikace označuje jako externí tržní kurzy, nikoli jako kurzy Tipsportu nebo Fortuny.

- Kurzy starší než 12 hodin se nepoužijí jako aktuální.
- Zápas se propojí jen při dostatečné shodě obou hráčů, turnaje a data; při nejasné shodě se kurz nezobrazí.
- Modelová pravděpodobnost a modelový férový kurz jsou oddělené od externích kurzů.
- Časové snímky se ukládají do `data/tennis_odds_log.csv`; stejné kurzy se stejnou aktualizací feedu se při opakovaném běhu neduplikují. Po dostupnosti výsledku se doplní vítěz.
- Zdroj je komunitní a může být dočasně nedostupný. V takovém případě zůstává tenisový model funkční a dashboard výslovně ukáže, že kurz není dostupný.


## Kalkulačka a moje fiktivní sázky (v7.0)
- Nahoře zadáš částku; u každého zápasu (fotbal i tenis) se u kurzů ukáže možná výhra. Klepnutím na kurz si sázku zapíšeš (kurz lze přepsat na ten, který bys opravdu dostal).
- Záložka **💰 Moje sázky**: zisk/ztráta, ROI, úspěšnost, graf, seznam, export/import (JSON, CSV). Sázky se po odehrání zápasu vyhodnotí samy z dat systému, jinak je lze označit ručně.
- Data jsou uložená **jen v prohlížeči** (localStorage). Při vymazání dat prohlížeče zmizí, proto občas použij Export.
- Modelové (férové) kurzy nejsou nabídka sázkovky; sázky na ně se počítají zvlášť jako fiktivní.
- Tenis má podmenu Zápasy / Vyhodnocení / Backtest a filtr podle ATP, WTA a turnaje.
