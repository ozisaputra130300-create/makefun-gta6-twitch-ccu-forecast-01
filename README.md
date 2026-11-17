# makefun-gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ozisaputra130300-create/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ozisaputra130300-create/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many concurrent Twitch viewers watch Grand Theft Auto VI in each hour 2–24 of its release day (2026-11-19 UTC), as five quantiles per hour. `main.ts` (Node 22, standard library only) reads the case file, fits its model at runtime from the comparables inside that file, and writes one `<id>.out` per case.

## The model

GTA VI unlocks at midnight local time, country by country, so Twitch viewership does not stay flat through the UTC day: it dips in the small hours (the US is asleep) and ramps into the evening. Persistence (every hour = hour 1) misses that shape, which is why the house baseline loses about 0.387 on the practice.

The key fact the comparables show: the ratio `viewers[h] / viewers[1]` depends strongly on hour 1's size. A launch whose hour 1 is already huge (875k for ELDEN RING) stays near flat (about 0.5x at the midday dip, about 1.0x in the evening); a launch whose hour 1 is tiny (164 for Silksong) explodes 100–1000x by evening. The log correlation of hour-1 size with the later ratio is about −0.93. So the forecast is conditioned on hour 1's print.

Two estimators, blended in log space (their geometric mean), then lightly smoothed across neighbouring hours:

- **kNN ratios (K=8):** the 8 comparables whose log hour-1 count is closest to the case's, and the empirical quantiles of their ratios at the five levels.
- **Regression + local residuals (K=12):** per hour, an OLS fit of log viewers[h] on log viewers[1] across the comparables (the slope falls from about 1.0 at hour 2 to about 0.06 at hours 19–24 — the evening peak barely depends on hour 1), plus the empirical quantiles of the fit's residuals from the 12 comparables nearest in log hour-1.

If a case file carries fewer than 8 usable comparables, a baked-in regression with global residual quantiles (fit offline on 37 past launches) is used instead.

## Validation (offline)

Answers for the four practice launches taken from SullyGnome's hourly Twitch category averages (hour 1 matches the case files exactly). Mean pinball loss on the log scale:

| model | practice mean |
|---|---|
| house baseline (persistence) | 0.387 |
| this entry (smoothed blend) | **0.220** |

Per case: Wolverine 0.320, Red Dead Redemption 2 0.207, God of War Ragnarök 0.156, Spider-Man 2 0.199. Leave-one-out over the 37 comparables: baseline 0.930, this entry 0.396. Typical miss about 45%, 90% coverage about 0.89 — an honest spread. The program is deterministic (no clock, no network, no randomness) and runs all four practice cases in about 0.3 s, far under the 8 s cap per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=sy0tw1m4&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
