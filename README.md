# FPL HoldPlanner DK

Et dansk, statistikbaseret beslutningsværktøj til Fantasy Premier League. Projektet er under en kontrolleret genopbygning til 2026/27-sæsonen: målet er reproducerbare data, ærlige prognoser, backtests og lovlige optimeringsforslag — ikke sorte bokse eller automatiske transfers.

## Status

Følgende fundament er implementeret:

- centraliserede FPL-regler for 2026/27 med tests
- en præcis MILP-optimering af et nyt 15-mandshold uden manager-ID
- en global, fortløbende gameweek-horisont, som bevarer blanks og doubles korrekt
- snapshots med metadata og atomisk skrivning til senere backtests
- adapter til Solio Analytics' offentlige JSON-feed
- korrekt brug af FPL's `selling_price` for ejede spillere
- tidsbegrænset Streamlit-cache oven på HTTP-kald med retry/backoff

Den nuværende interne pointmodel er fortsat en **eksperimentel heuristisk baseline**. Dens output må ikke læses som validerede prognoser, før walk-forward-backtests og kalibrering er på plads. AI- og chipråd er derfor sat på pause som beslutningsmotorer.

## Kør lokalt

Kræver Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r fpl_app/requirements.txt -r requirements-dev.txt
streamlit run fpl_app/app.py
```

Test fundamentet:

```bash
python -m pytest
python -m compileall -q fpl_app
```

Gem et valideret Solio-snapshot til det git-ignorerede point-in-time-lager:

```bash
PYTHONPATH=fpl_app python scripts/fetch_solio_snapshot.py
```

Kopiér ved behov `fpl_app/secrets.toml.example` til `fpl_app/.streamlit/secrets.toml`. Hemmeligheder må aldrig committes.

## Deploy på Vercel

Repoet indeholder `Dockerfile.vercel`, så Vercel kan bygge den eksisterende
Streamlit-app som en containerfunktion uden at omskrive brugerfladen. Opret et
Vercel-projekt fra GitHub-repoet; hver commit bygger et nyt image automatisk.

Til privat brug på Hobby-planen skal **Vercel Authentication / Standard
Protection** aktiveres, og den beskyttede preview- eller deployment-URL skal
bruges. Den korte produktionsadresse er ikke omfattet af Standard Protection.
Undlad derfor at dele eller bruge produktionsadressen, medmindre der tilføjes
applikationslogin eller betalt beskyttelse af alle deployments.

Containeren bruger Vercel Functions og WebSockets. På Hobby-planen kan en aktiv
Streamlit-session blive afbrudt ved funktionens maksimale varighed; appen kan
genindlæses uden at foretage handlinger i FPL.

## Datakilder

Se også den konkrete [datakilde- og købsguide](docs/DATA_SOURCES.md).

| Kilde | Brug | Adgang/licensprincip |
|---|---|---|
| [Officiel FPL API](https://fantasy.premierleague.com/api/bootstrap-static/) | spillere, priser, status, fixtures og managerdata | offentlig endpoint; snapshot rå respons med hentetid |
| [Solio Analytics](https://fpl.solioanalytics.com/) | ekstern projektion og hold-/kampestimater | offentligt JSON-feed; vis attribution “Solio Analytics” |
| Oddsleverandør | markedssandsynligheder | valgfri API-nøgle; rådata må ikke publiceres uden licens |
| Betalte projektioner | senere model-ensemble/benchmark | kun brugerens egen eksport; ingen redistribuering i repoet |

Kildeadaptere skal bevare navn, URL, hentetid, sæson/gameweek, skemaversion og checksum. Snapshotfiler og købte CSV'er er runtime-data og hører ikke hjemme i Git.

## Arkitektur

```text
FPL/Solio/odds -> adaptere -> validerede snapshots -> features/prognoser
                                                     -> holdoptimering
                                                     -> senere backtests
                                                     -> Streamlit-visning
```

- `fpl_app/domain/`: sæsonregler og kildekontrakter
- `fpl_app/services/`: API-adaptere, cache og snapshots
- `fpl_app/logic/`: prognose-baseline og optimering
- `fpl_app/pages/`: Streamlit-visninger
- `tests/`: enheds- og kontrakttests

## Næste milepæle

1. Gem deadline-snapshots hver gameweek og byg et point-in-time træningssæt uden datalækage.
2. Walk-forward-backtest mod simple baselines og eksterne projektioner; rapportér MAE, calibration, captain regret og transfer regret.
3. Erstat heuristikken med et kalibreret ensemble for point og spilletid.
4. Udvid optimeringen fra starttrup til flerugers transfers, frie transfers, hits, chips, bench og captaincy.
5. Aktivér automatiske ugentlige rapporter efter deadline-/tilgængelighedstests — men lad alle FPL-handlinger være manuelle.

## Sikkerhed og ansvar

Appen skal ikke modtage dit FPL-password eller foretage transfers. Et manager-ID er offentligt og bruges kun til at læse holddata. Prognoser er usikre estimater, og brugeren skal altid godkende den endelige beslutning i FPL.
