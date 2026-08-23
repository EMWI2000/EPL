# FPL HoldPlanner DK

Et dansk, statistikbaseret beslutningsværktøj til Fantasy Premier League. Den primære app er bygget til Vercel med en responsiv Next.js-brugerflade og en stateless Python-beregningsfunktion. Målet er reproducerbare data, usikkerhedsmarkerede prognoser, backtests og regelgyldige holdforslag. Appen foretager ikke transfers.

## Status

Følgende er implementeret:

- Vercel-native Next.js-app uden Streamlit-sessioner eller WebSockets
- privat GitHub-login med en tilladelsesliste med et uforanderligt GitHub-bruger-id
- beskyttet backend-for-frontend; den tunge Python-funktion kan ikke kaldes direkte uden en intern nøgle
- centraliserede FPL-regler for 2026/27 med tests
- en minutmodel, der kombinerer FPL-status, antal starter og historiske minutter
- empirical-Bayes-shrinkage mod dynamiske positions- og prispriorer med 900 minutters priorstyrke
- pointdekomponering, forventede minutter og usikkerhed for hver spiller og gameweek
- en lovlig 15-mandstrup med separat XI, kaptajn, vicekaptajn og bænk i hver gameweek
- en fast shortlist på 45 spillere til serverless-kørslen, som dækker billige budgetspillere, værdi og GW-specialister
- MILP-optimering af truppen, XI og kaptajn inden for shortlisten; bænken sorteres efter tilgængelighedsjusteret EP
- en global, fortløbende gameweek-horisont, som bevarer blanks og doubles korrekt
- snapshots med metadata og atomisk skrivning til senere backtests
- en ugeplanlægger, der indlæser det senest offentliggjorte managerhold og kræver bekræftelse af bank, frie transfers, priser og chipstatus
- rullende transferanalyse med hold, én eller flere transfers, hits, halv prisgevinst ved salg og højst fem frie transfers efter FPL-reglerne
- en afgrænset 8-GW strategimotor med fire sammenkædede deadlines, som fører trup, bank, frie transfers, hits og salgsprisbasis videre
- personlige chipscenarier baseret på officiel chiphistorik: Wildcard-genopbygning, Free Hit-screening samt marginalværdi for Bench Boost og Triple Captain
- en frivillig beslutningsjournal, der kun gemmer et kompakt, valideret deadline-resumé i brugerens egen browser
- et eksplicit AI-deadlinebrief med GPT-5.6 Sol, meget høj reasoning, kvalitativ research og et betinget flerugersperspektiv
- korrekt brug af FPL's `selling_price` for ejede spillere
- lækagesikre deadline-folds, evalueringsmetrics og GW-parret bootstrap
- en ML-pipeline med fælles featurekontrakt til træning og drift; modelartefakter er deaktiveret, indtil de er valideret
- gennemsigtig visning af projektion, kilde, xMin, usikkerhed, start-XI, kaptajn, bænk og datadækning

V2 er standard i webappen. Legacy-baselinen kan vælges til sammenligning. Begge er **eksperimentelle**. De tekniske tests dokumenterer beregningerne, men dokumenterer endnu ikke prognosekvaliteten. V2 skal derfor walk-forward-testes og kalibreres på deadline-snapshots, før den kan kaldes valideret. Chipscenarierne er afgrænsede beslutningsstøtter og ikke et dokumenteret sæsonoptimum.

### Ugeplanlægning

Ugeplanlæggeren tager udgangspunkt i managerens senest offentliggjorte hold. FPL viser ikke igangværende transfers eller et ventende chipvalg før næste deadline, så synkroniseringen skal ske, før ugens handlinger udføres. Officiel chiphistorik bruges til at udlede, hvilke chips der er tilbage i den aktuelle sæsonhalvdel. Brugeren bekræfter derefter bank, antal frie transfers, aktuelle købs- og salgspriser og at ingen ny chip allerede er aktiveret. Appen foretager aldrig transfers eller chipaktiveringer.

Når `FPL_MANAGER_ID` er konfigureret, hentes dette hold automatisk efter login, og anbefalingsruten afviser andre eller manglende manager-ID'er. Den generelle funktion til at bygge en ny trup skjules i den bundne produktionsapp, så alle viste anbefalinger tager udgangspunkt i det konfigurerede managerhold.

Planlæggeren sammenligner at rulle transferen med forskellige transfers til næste deadline. Hver plan vurderes over den valgte prognosehorisont. Den medregner transferhits, bænkens forventede bidrag ved udeblivelser og de særlige salgsprisregler. For transferantal 1-2 kan den vise flere alternativer; for 3-5 beregner den én bedst plan pr. antal.

Ved en personlig ugeanalyse beregner appen desuden en separat standardhorisont på otte gameweeks. Ét MILP vælger første skridt blandt de allerede validerede og viste handlinger og modellerer derefter tre foreløbige deadlines med højst to transfers pr. deadline. Det valgte første skridt bliver appens kanoniske handling, så topboks, opstilling, AI-review og beslutningsjournal bruger samme plan. Resten af vinduet vurderes med truppen efter deadline fire. Modellen medfører bank, FT, hits og den korrekte oprindelige salgsprisbasis; spillere, der købes senere, får dagens pris som ny basis. Priser og information holdes ellers faste. Resultatet er bevist optimalt inden for det afgrænsede kandidatfelt og disse grænser, ikke globalt optimalt, og alle fremtidige skridt skal genberegnes ved den virkelige deadline.

På samme roadmap sammenlignes fire chipscenarier. Wildcard løser én fuld permanent 15-mands genopbygning mod planen uden chip. Triple Captain måles som én ekstra forventet kopi af roadmapets anførerpoint, mens Bench Boost måles som bænkens merpoint efter fradrag for normal forventet indskiftningsværdi. Free Hit løses kun som en midlertidig trup, når den officielle fixturekalender allerede viser en blank eller double; ellers anbefaler screeningen at gemme chippen. Et løst Free Hit forbliver et overvågningspunkt, indtil den permanente transferplan efter chippen også kan genoptimeres. Scenarierne vises som hold, overvåg eller overvej og aktiveres aldrig af appen. Hvis den lange solver eller et chipscenarie ikke når et bevist resultat inden det fælles serverless-tidsbudget, bevares den almindelige næste-deadline-anbefaling. To-deadline-modellen fungerer fortsat som fallback.

Snapshot-checksummen identificerer den synkroniserede tilstand, men serveren kontrollerer ikke, om managerens private kladde siden er ændret.

Efter kontrol kan den viste anbefaling gemmes i en lokal beslutningsjournal. Den indeholder kun deadline, modelversion, bekræftet bank/FT, den valgte solver- eller AI-handling, kompakte transfers samt den valgte plans kaptajn. Manager-ID, GitHub-identitet, tokens, rå AI-output og hele API-svaret gemmes ikke. Journalen sendes ikke tilbage til beregningen eller OpenAI og kan eksporteres eller slettes fra browseren.

### AI-kvalificering til næste deadline

Efter en ugeplan er beregnet, kan brugeren aktivt bestille en second opinion. Next.js sender en stramt afgrænset fodboldkontekst til OpenAI Responses API: bred ranggruppe, bekræftet bank og frie transfers, solverens bedste plan og alternativer, et kompakt roadmap og chipscenarier, start-XI, kaptajn, projektioner, minutter, usikkerhed og prissignaler. GitHub-identitet, manager-ID, snapshot-checksum, cookies, beslutningsjournal og tokens fjernes, før API-kaldet foretages.

Revieweren kører som standard med GPT-5.6 Sol og `xhigh` reasoning. Den laver frisk webresearch på tilladte Premier League-, BBC- og relevante officielle klubdomæner og strukturerer holdnyt, taktisk rolle, minutter, dødbolde, kampprogram og prisrisiko som kvalitative signaler. Den kan kun bekræfte bedste plan, anbefale at vente på konkret information eller vælge et nummereret solver-alternativ. Roadmap og chipscenarier ejes af solverlaget: AI'en må forklare og kvalificere dem, men må ikke opfinde transfers, ændre chipscenariet eller præsentere foreløbige skridt som låste handlinger. Kilder vises som klikbare links. Kaldet er manuelt for at styre omkostninger og kan tage flere minutter ved `xhigh`; en fejl skjuler aldrig den deterministiske anbefaling. `store: false` er aktiveret; OpenAI kan fortsat behandle API-indhold efter kontoens gældende data- og retentionvilkår.

## Kør lokalt

Kræver Node.js 20.9+ og Python 3.12.

```bash
npm ci
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env.local
```

`npm run dev` starter Next.js-delen. Brug `npx vercel dev`, når både Next.js- og Python-ruterne skal køre lokalt i samme miljø. Et lokalt GitHub OAuth App-callback skal da være `http://localhost:3000/api/auth/callback/github`.

Kvalitetskontrol:

```bash
npm run typecheck
npm run build
python -m pytest
python -m compileall -q api fpl_app
```

Den tidligere Streamlit-app ligger fortsat i `fpl_app/` som reference og kan køres med dens separate requirements-fil. Hemmeligheder må aldrig checkes ind.

## Deploy på Vercel

Forbind GitHub-repoet direkte til et Vercel-projekt. Vercel registrerer Next.js,
bygger frontend og pakker `api/compute.py` og `api/manager_state.py` som separate Python Functions. Push
til den valgte produktionsbranch udløser derefter automatisk deployment.

Følgende miljøvariabler skal oprettes i Vercel og må aldrig gemmes i GitHub:

| Variabel | Formål |
|---|---|
| `BETTER_AUTH_URL` | Appens kanoniske `https://...vercel.app`-adresse |
| `BETTER_AUTH_SECRET` | mindst 32 tilfældige bytes til krypterede sessions |
| `GITHUB_CLIENT_ID` | Client ID fra GitHub OAuth App |
| `GITHUB_CLIENT_SECRET` | Client secret fra GitHub OAuth App |
| `ALLOWED_GITHUB_ID` | numerisk GitHub-ID, aktuelt `199608244` |
| `FPL_MANAGER_ID` | offentligt FPL entry-ID, som automatisk synkroniseres efter login |
| `INTERNAL_API_TOKEN` | mindst 32 tilfældige bytes mellem Next.js og Python |
| `OPENAI_API_KEY` | server-side projektnøgle til det valgfrie AI-deadlinebrief; deploymenten accepterer også aliaset `FANTASY` |
| `OPENAI_MODEL` | Responses-model; eneste tilladte produktionsmodel er `gpt-5.6-sol` |
| `OPENAI_REASONING_EFFORT` | reasoning-niveau: `high`, `xhigh` (standard) eller `max` |
| `SESSION_VERSION` | start med `1`; hæv værdien for at logge alle sessioner ud |

GitHub OAuth App skal have produktionsadressen som Homepage URL og
`https://<produktionsdomæne>/api/auth/callback/github` som callback. App-login
beskytter også den stabile produktionsadresse, som Vercels gratis Standard
Protection ikke dækker.

## Datakilder

Se også den konkrete [datakilde- og købsguide](docs/DATA_SOURCES.md).

| Kilde | Brug | Adgang/licensprincip |
|---|---|---|
| [Officiel FPL API](https://fantasy.premierleague.com/api/bootstrap-static/) | spillere, priser, status, fixtures og managerdata | offentligt endpoint; snapshot rå respons med hentetid |
| Solio Analytics | ikke anvendt af Vercel-appen | Vercel-appen henter ikke Solio-data automatisk; en lovlig, brugerleveret eksport kan senere bruges som benchmark |
| Oddsleverandør | markedssandsynligheder | valgfri API-nøgle; rådata må ikke publiceres uden licens |
| Betalte projektioner | senere model-ensemble/benchmark | kun brugerens egen eksport; ingen redistribuering i repoet |

Kildeadaptere skal bevare navn, URL, hentetid, sæson/gameweek, skemaversion og checksum. Snapshotfiler og købte CSV'er er runtime-data og hører ikke hjemme i Git.

## Arkitektur

```text
Browser -> GitHub-login -> Next.js BFF
                             |-> intern token -> Python Functions -> officiel FPL API
                             |                                  -> prognoser + MILP
                             |                                  -> JSON -> Next.js UI
                             `-> OpenAI Responses API + afgrænset webresearch
```

- `app/` og `components/`: Next.js UI, login og beskyttet BFF
- `api/`: stateless Vercel Python Functions
- `lib/`: auth- og sessionsgrænse
- `fpl_app/domain/`: sæsonregler og kildekontrakter
- `fpl_app/services/`: API-adaptere, cache og snapshots
- `fpl_app/logic/`: forecast v2, legacy-baseline og flerugersoptimering
- `fpl_app/evaluation/`: deadline-folds, metrics og bootstrap
- `fpl_app/ml/v2_dataset.py`: fælles point-in-time-featurebygning til træning og serving
- `fpl_app/pages/`: tidligere Streamlit-visninger, bevaret som reference
- `tests/`: enheds- og kontrakttests

## Næste milepæle

1. Gem deadline-snapshots hver gameweek, så forecast v2 kan evalueres uden datalækage.
2. Kør walk-forward-backtest mod legacy og eventuelle lovligt indsamlede benchmarks. Rapportér MAE, RMSE, kalibrering, Brier-score og captain regret.
3. Kalibrér minut- og tilgængelighedsmodellen. Justér priorstyrke og shortlist ud fra beslutningsregret.
4. Sæt kun et ML-modelartefakt i drift, hvis det slår de simple baselines på uafhængige gameweeks.
5. Evaluer roadmap- og chipscenarierne mod gemte deadlines, herunder transferregret, chipregret og robusthed over for pris- og minutændringer.
6. Udvid først fra fire modellerede deadlines til komplette sæsonsekvenser, når runtime, datagrundlag og walk-forward-backtests kan bære det. Alle FPL-handlinger forbliver manuelle.

## Sikkerhed og ansvar

Appen skal ikke modtage din FPL-adgangskode eller foretage transfers. Et manager-ID er offentligt og bruges kun til at læse holddata. Prognoser er usikre estimater, og brugeren skal altid godkende den endelige beslutning i FPL.
