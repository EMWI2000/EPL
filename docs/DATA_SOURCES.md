# Datakilder og købsguide

Denne prioritering er lavet til et personligt FPL-beslutningssystem. En kilde er kun værdifuld, hvis dens data kan gemmes før deadline, matches stabilt til FPL-spillere og må bruges til det aftalte formål.

## Anbefalet rækkefølge

| Prioritet | Kilde | Formål | Beslutning nu |
|---|---|---|---|
| 1 | [Officiel FPL](https://fantasy.premierleague.com/api/bootstrap-static/) | priser, positioner, status, fixtures, managerdata og facit | brug altid; snapshot før deadline |
| 2 | Brugerleveret projectionseksport | ekstern GW-projektion som benchmark | brug kun en eksport, som licensen tillader; Vercel-appen henter ikke Solio-data automatisk |
| 3 | [FPL Review](https://fplreview.com/) | betalte projektioner og planlægningsværktøjer | køb én måned som benchmark, hvis eksport og vilkår passer |
| 4 | [Fantasy Football Scout](https://www.fantasyfootballscout.co.uk/) | medlemstal, forventede opstillinger, pressemøder og kvalitativ kontekst | køb hvis availability/minutes er den største modelsvaghed |
| 5 | [The Odds API](https://the-odds-api.com/) | konsistente markedssandsynligheder | start på gratis niveau; betal kun efter dokumenteret backtestløft |
| 6 | Opta/Stats Perform, StatsBomb eller SkillCorner | event-, tracking- og avancerede eventdata | vent; pris og licens er normalt overkill til fase 1 |

Abonnementer er ikke en erstatning for snapshots. En flot aktuel projektion kan ikke backtestes korrekt, hvis den overskrives efter deadline.

## Købstest før et abonnement

1. Kan data eksporteres maskinlæsbart med spiller-ID eller stabilt navn/hold/position?
2. Må eksporten bruges i et privat analyseværktøj og gemmes historisk?
3. Er publikation, deling og modeltræning tilladt eller særskilt licenseret?
4. Findes der tidsstempel, gameweek og tydelig betydning af hver kolonne?
5. Kan en måneds snapshots sammenlignes med gratis baselines før årsbetaling?

Købte projektioner og API-svar må ikke committes eller redistribueres uden udtrykkelig licens. Repoet bør kun indeholde adapterkode, skemabeskrivelser og syntetiske testfixtures.

## Gode tekniske referenceprojekter

- [OpenFPL](https://github.com/sertalpbilal/FPL-Optimization-Tools): optimeringsværktøjer og praktiske FPL-workflows.
- [AIrsenal](https://github.com/alan-turing-institute/AIrsenal): reproducerbar FPL-modellering og optimering over flere gameweeks.
- [open-fpl-solver](https://github.com/solioanalytics/open-fpl-solver): solver-arkitektur og constraints.

De bruges som designreferencer, ikke som facit. Egen backtest skal afgøre, hvilke features og beslutningsregler der virker på 2026/27-data.

## Minimumsmetadata pr. snapshot

- `source_id` og original URL
- observeret hentetid i UTC
- kildens effektive/genererede tidspunkt
- sæson og gameweek
- skemaversion
- SHA-256 og byteantal
- licens-/attributionsnote

Det gør det muligt senere at bevise, hvad modellen faktisk vidste ved deadline.
