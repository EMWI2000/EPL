# FPL HoldPlanner DK

Et dansk, statistikbaseret beslutningsværktøj til Fantasy Premier League. Den primære app er bygget til Vercel med en responsiv Next.js-brugerflade og en stateless Python-beregningsfunktion. Målet er reproducerbare data, ærlige prognoser, backtests og lovlige optimeringsforslag — ikke sorte bokse eller automatiske transfers.

## Status

Følgende fundament er implementeret:

- Vercel-native Next.js-app uden Streamlit-sessioner eller WebSockets
- privat GitHub-login med allowlist på en uforanderlig GitHub-bruger-ID
- beskyttet backend-for-frontend; den tunge Python-funktion kan ikke kaldes direkte uden en intern nøgle
- centraliserede FPL-regler for 2026/27 med tests
- en præcis MILP-optimering af et nyt 15-mandshold uden manager-ID
- en global, fortløbende gameweek-horisont, som bevarer blanks og doubles korrekt
- snapshots med metadata og atomisk skrivning til senere backtests
- adapter til Solio Analytics' offentlige JSON-feed
- korrekt brug af FPL's `selling_price` for ejede spillere
- gennemsigtig visning af projektion, kilde, start-XI, kaptajn, bænk og datadækning

Den nuværende interne pointmodel er fortsat en **eksperimentel heuristisk baseline**. Dens output må ikke læses som validerede prognoser, før walk-forward-backtests og kalibrering er på plads. AI- og chipråd er derfor sat på pause som beslutningsmotorer.

## Kør lokalt

Kræver Node.js 20.9+ og Python 3.12.

```bash
npm ci
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env.local
```

`npm run dev` starter Next.js-delen. Brug `npx vercel@59.1.4 dev`, når både Next.js- og Python-ruterne skal køre lokalt i samme miljø. Et lokalt GitHub OAuth App-callback skal da være `http://localhost:3000/api/auth/callback/github`.

Kvalitetskontrol:

```bash
npm run typecheck
npm run build
python -m pytest
python -m compileall -q api fpl_app
```

Gem et valideret Solio-snapshot til det git-ignorerede point-in-time-lager:

```bash
PYTHONPATH=fpl_app python scripts/fetch_solio_snapshot.py
```

Den tidligere Streamlit-app ligger fortsat i `fpl_app/` som reference og kan køres med dens separate requirements-fil. Hemmeligheder må aldrig committes.

## Deploy på Vercel

Forbind GitHub-repoet direkte til et Vercel-projekt. Vercel registrerer Next.js,
bygger frontend og pakker `api/compute.py` som en separat Python Function. Push
til den valgte production branch udløser derefter automatisk deployment.

Følgende miljøvariabler skal oprettes i Vercel — aldrig i GitHub:

| Variabel | Formål |
|---|---|
| `BETTER_AUTH_URL` | Appens kanoniske `https://...vercel.app`-adresse |
| `BETTER_AUTH_SECRET` | mindst 32 tilfældige bytes til krypterede sessions |
| `GITHUB_CLIENT_ID` | Client ID fra GitHub OAuth App |
| `GITHUB_CLIENT_SECRET` | Client secret fra GitHub OAuth App |
| `ALLOWED_GITHUB_ID` | numerisk GitHub-ID, aktuelt `199608244` |
| `INTERNAL_API_TOKEN` | tilfældig intern nøgle mellem Next.js og Python |
| `SESSION_VERSION` | start med `1`; hæv værdien for at logge alle sessioner ud |

GitHub OAuth App skal have produktionsadressen som Homepage URL og
`https://<produktionsdomæne>/api/auth/callback/github` som callback. App-login
beskytter også den stabile produktionsadresse, som Vercels gratis Standard
Protection ikke dækker.

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
Browser -> GitHub-login -> Next.js BFF -> intern token -> Python Function
                                                   -> FPL/Solio-adaptere
                                                   -> prognoser + MILP
                                                   -> JSON -> Next.js UI
```

- `app/` og `components/`: Next.js UI, login og beskyttet BFF
- `api/`: små stateless Vercel Python Functions
- `lib/`: auth- og sessionsgrænse
- `fpl_app/domain/`: sæsonregler og kildekontrakter
- `fpl_app/services/`: API-adaptere, cache og snapshots
- `fpl_app/logic/`: prognose-baseline og optimering
- `fpl_app/pages/`: tidligere Streamlit-visninger, bevaret som reference
- `tests/`: enheds- og kontrakttests

## Næste milepæle

1. Gem deadline-snapshots hver gameweek og byg et point-in-time træningssæt uden datalækage.
2. Walk-forward-backtest mod simple baselines og eksterne projektioner; rapportér MAE, calibration, captain regret og transfer regret.
3. Erstat heuristikken med et kalibreret ensemble for point og spilletid.
4. Udvid optimeringen fra starttrup til flerugers transfers, frie transfers, hits, chips, bench og captaincy.
5. Aktivér automatiske ugentlige rapporter efter deadline-/tilgængelighedstests — men lad alle FPL-handlinger være manuelle.

## Sikkerhed og ansvar

Appen skal ikke modtage dit FPL-password eller foretage transfers. Et manager-ID er offentligt og bruges kun til at læse holddata. Prognoser er usikre estimater, og brugeren skal altid godkende den endelige beslutning i FPL.
