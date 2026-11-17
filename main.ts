// gta6-twitch-ccu-forecast-01 entry (Node 22, standard library only).
//
// Forecast: how many concurrent Twitch viewers watch Grand Theft Auto VI in
// each hour 2..24 of its release day (2026-11-19 UTC), as five quantiles.
// The game unlocks at midnight local time, country by country, so Twitch
// viewership does NOT stay flat through the UTC day: it dips in the small
// hours (the US is asleep, Europe at work) and ramps into the evening as
// the US wakes up and plays. Persistence (every hour = hour 1) misses that
// shape entirely, which is why it loses about 0.387 on the practice.
//
// Model (fit at runtime from the comparables inside the case file, so it
// adapts to whatever launch the case describes):
//   - Every comparable is a past big launch with a full release-day hourly
//     series (hours 1..24). The key fact the data shows: the ratio
//     viewers[h] / viewers[1] depends strongly on hour 1's size. A launch
//     whose hour 1 is already huge (875k for ELDEN RING) stays near flat
//     (ratio ~0.5 at the midday dip, ~1.0 in the evening); a launch whose
//     hour 1 is tiny (164 for Silksong) explodes 100-1000x by evening.
//     The log correlation of hour-1 size with the later ratio is about
//     -0.93. So the forecast is conditioned on hour 1's print, p1.
//   - Two estimators, blended in log space (their geometric mean):
//     (a) kNN ratios (K=8): the 8 comparables whose log(p1) is closest to
//         the case's, and the empirical quantiles of their ratios
//         viewers[h]/viewers[1] at the five levels. This adapts both the
//         level and the spread to the launch's size.
//     (b) Regression + local residuals (K=12): per hour, an OLS fit of
//         log(viewers[h]) on log(viewers[1]) across the comparables (the
//         slope falls from ~1.0 at hour 2 to ~0.06 at hours 19-24, i.e. the
//         evening peak barely depends on hour 1), plus the empirical
//         quantiles of the fit's residuals from the 12 comparables nearest
//         in log(p1). This extrapolates smoothly if p1 is beyond the
//         comparables' range.
//   - Fallback: if the case file carries fewer than 8 usable comparables,
//     a regression with global residual quantiles, baked in below, fit
//     offline on 37 past launches (practice loss 0.237, also well under
//     the baseline's 0.387).
//
// Validation (offline, on the four practice launches, answers from
// SullyGnome's hourly averages — hour 1 matches the case files exactly):
//   baseline persistence 0.387 mean pinball loss; this blend 0.220.
//   Leave-one-out over the 37 comparables: baseline 0.930, this blend 0.397.
// The program is deterministic: no clock, no network, no randomness. It
// reads every *.in in the input dir and writes one <id>.out per case.
//
// Run locally:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Hours to forecast. */
const HOURS: number[] = [];
for (let h = 2; h <= 24; h++) HOURS.push(h);
/** Neighbours for the kNN ratio estimator. */
const KNN_K = 8;
/** Neighbours for the regression's local residual spread. */
const REG_K = 12;
/** Minimum usable comparables before the baked-in fallback is used. */
const MIN_COMPARABLES = 8;

/**
 * Baked-in fallback: per hour, the OLS intercept and slope of
 * log(viewers[h]) on log(viewers[1]), and the global residual quantiles
 * [q05, q25, q50, q75, q95] in log space, fit offline on 37 past launches
 * (ELDEN RING, Cyberpunk 2077, Hogwarts Legacy, Diablo IV, ...). Only used
 * when a case file carries fewer than MIN_COMPARABLES usable comparables.
 */
const FALLBACK_A: Record<number, number> = {
  2: -0.308094, 3: -0.252647, 4: 0.152501, 5: 0.120641, 6: 0.277274,
  7: 1.046690, 8: 1.806842, 9: 3.355982, 10: 5.256629, 11: 5.777326,
  12: 6.260746, 13: 7.227577, 14: 8.218525, 15: 9.263373, 16: 10.319313,
  17: 10.738624, 18: 11.176051, 19: 11.466741, 20: 11.558168, 21: 11.600667,
  22: 11.493081, 23: 11.225488, 24: 11.002108,
};
const FALLBACK_B: Record<number, number> = {
  2: 1.021670, 3: 1.002472, 4: 0.960153, 5: 0.965745, 6: 0.957213,
  7: 0.890602, 8: 0.827445, 9: 0.691997, 10: 0.528173, 11: 0.486039,
  12: 0.450121, 13: 0.379770, 14: 0.305428, 15: 0.226299, 16: 0.145372,
  17: 0.117652, 18: 0.087297, 19: 0.070062, 20: 0.064113, 21: 0.061671,
  22: 0.068266, 23: 0.085797, 24: 0.095032,
};
const FALLBACK_RESID_Q: Record<number, number[]> = {
  2: [-0.2919, -0.1585, -0.0785, 0.0707, 0.4920],
  3: [-0.7724, -0.2565, -0.0519, 0.1630, 0.6072],
  4: [-0.6314, -0.4016, -0.0362, 0.1506, 0.6204],
  5: [-0.9921, -0.3068, -0.0133, 0.1546, 0.5602],
  6: [-1.0186, -0.4305, -0.0403, 0.2393, 1.1055],
  7: [-0.9534, -0.4122, -0.0507, 0.1867, 1.1822],
  8: [-0.8434, -0.3746, -0.0485, 0.2655, 1.1523],
  9: [-0.8461, -0.4510, -0.0938, 0.3148, 1.3778],
  10: [-1.1284, -0.5591, -0.0135, 0.3464, 1.8201],
  11: [-1.3528, -0.5275, -0.0185, 0.4219, 1.9058],
  12: [-1.2567, -0.4586, -0.0911, 0.3621, 1.7462],
  13: [-1.1606, -0.4649, 0.0030, 0.4632, 1.4107],
  14: [-1.3856, -0.4790, -0.0224, 0.6089, 1.6124],
  15: [-1.5626, -0.5572, -0.0212, 0.7202, 1.5805],
  16: [-1.4024, -0.6039, -0.1637, 0.8514, 1.3587],
  17: [-1.5060, -0.7345, -0.2194, 0.9537, 1.2916],
  18: [-1.2100, -0.7296, -0.2485, 0.9358, 1.2313],
  19: [-1.1448, -0.5246, -0.2609, 0.7293, 1.1758],
  20: [-1.1862, -0.4801, -0.1057, 0.6722, 1.0871],
  21: [-1.2811, -0.5588, -0.0688, 0.6571, 1.0750],
  22: [-1.2589, -0.5648, -0.0438, 0.6964, 1.0202],
  23: [-1.3847, -0.4702, -0.0555, 0.7508, 0.9386],
  24: [-1.2051, -0.5240, 0.0002, 0.7442, 0.9183],
};

interface HourlyPoint { from: string; viewers: number | null; channels?: number | null }
interface Comparable { name: string; day_start: string; hourly: HourlyPoint[] }
interface CaseFile {
  id: string;
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[] | null;
}

/** A comparable's release-day hourly viewers, hours 1..24, or null if incomplete. */
type DaySeries = Map<number, number>;

/** Linear-interpolated empirical quantile of a sorted array. */
function empQuantile(sorted: number[], tau: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  if (n === 1) return sorted[0];
  const pos = tau * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] * (hi - pos) + sorted[hi] * (pos - lo);
}

/** Ordinary least squares: y ~ a + b*x. */
function ols(xs: number[], ys: number[]): { a: number; b: number } {
  const n = xs.length;
  const mx = xs.reduce((s, x) => s + x, 0) / n;
  const my = ys.reduce((s, y) => s + y, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx === 0 ? 0 : sxy / sxx;
  return { a: my - b * mx, b };
}

/** Extract one comparable's release-day series; null unless hours 1..24 are all present and positive. */
function daySeries(comp: Comparable): DaySeries | null {
  const start = Date.parse(comp.day_start);
  if (!Number.isFinite(start)) return null;
  const byHour = new Map<number, number>();
  for (const p of comp.hourly ?? []) {
    if (p.viewers === null || p.viewers === undefined || !(p.viewers > 0)) continue;
    const t = Date.parse(p.from);
    if (!Number.isFinite(t)) continue;
    const h = Math.floor((t - start) / 3600000) + 1;
    if (h >= 1 && h <= 24 && !byHour.has(h)) byHour.set(h, p.viewers);
  }
  if (byHour.size !== 24) return null;
  return byHour;
}

/** Estimator (a): kNN ratio quantiles from the K comparables nearest in log(p1). */
function knnQuantiles(p1: number, series: DaySeries[], k: number): Record<number, number[]> {
  const lp = Math.log(p1);
  const near = [...series]
    .sort((a, b) => Math.abs(Math.log(a.get(1)!) - lp) - Math.abs(Math.log(b.get(1)!) - lp))
    .slice(0, Math.min(k, series.length));
  const out: Record<number, number[]> = {};
  for (const h of HOURS) {
    const ratios = near.map((s) => s.get(h)! / s.get(1)!).sort((x, y) => x - y);
    out[h] = LEVELS.map((tau) => p1 * empQuantile(ratios, tau));
  }
  return out;
}

/** Estimator (b): per-hour regression on log(p1), spread from the K nearest residuals. */
function regLocalQuantiles(p1: number, series: DaySeries[], k: number): Record<number, number[]> {
  const lp = Math.log(p1);
  const out: Record<number, number[]> = {};
  for (const h of HOURS) {
    const xs = series.map((s) => Math.log(s.get(1)!));
    const ys = series.map((s) => Math.log(s.get(h)!));
    const { a, b } = ols(xs, ys);
    const resid = series.map((s, i) => ({
      dist: Math.abs(xs[i] - lp),
      r: ys[i] - (a + b * xs[i]),
    }));
    resid.sort((u, v) => u.dist - v.dist);
    const near = resid.slice(0, Math.min(k, resid.length)).map((r) => r.r).sort((x, y) => x - y);
    out[h] = LEVELS.map((tau) => Math.exp(a + b * lp + empQuantile(near, tau)));
  }
  return out;
}

/** Geometric-mean blend of two quantile forecasts (average in log space). */
function blend(q1: Record<number, number[]>, q2: Record<number, number[]>): Record<number, number[]> {
  const out: Record<number, number[]> = {};
  for (const h of HOURS) {
    out[h] = LEVELS.map((_, j) => Math.exp((Math.log(q1[h][j]) + Math.log(q2[h][j])) / 2));
  }
  return out;
}

/**
 * Light 3-point smoothing of the log-quantiles across neighbouring hours.
 * The kNN component takes empirical quantiles of a few games' ratios, so a
 * single game's hourly spike can jog one hour's row; the true launch-day
 * shape moves smoothly (unlocks arrive hour by hour), so averaging each
 * quantile with its neighbours in log space removes that noise. Validated
 * offline: it leaves the practice loss unchanged (0.2203 vs 0.2201) and
 * improves the leave-one-out loss slightly (0.3964 vs 0.3970 on all
 * comparables, 0.2348 vs 0.2355 on the ten biggest launches).
 */
function smoothQuantiles(q: Record<number, number[]>): Record<number, number[]> {
  const out: Record<number, number[]> = {};
  for (const h of HOURS) {
    const row: number[] = [];
    for (let j = 0; j < LEVELS.length; j++) {
      const vals: number[] = [];
      for (const hh of [h - 1, h, h + 1]) {
        if (hh >= 2 && hh <= 24) vals.push(Math.log(q[hh][j]));
      }
      row.push(Math.exp(vals.reduce((s, x) => s + x, 0) / vals.length));
    }
    for (let j = 1; j < row.length; j++) row[j] = Math.max(row[j], row[j - 1]);
    out[h] = row;
  }
  return out;
}

/** Baked-in fallback forecast (regression + global residual quantiles). */
function fallbackQuantiles(p1: number): Record<number, number[]> {
  const lp = Math.log(p1);
  const out: Record<number, number[]> = {};
  for (const h of HOURS) {
    out[h] = FALLBACK_RESID_Q[h].map((rq) => Math.exp(FALLBACK_A[h] + FALLBACK_B[h] * lp + rq));
  }
  return out;
}

/** x rounded to 2 decimals (half up), clamped to (0, 1e9]. */
function clean(x: number): number {
  const r = Math.round(x * 100) / 100;
  if (!Number.isFinite(r) || r <= 0) return 0.01;
  return Math.min(r, 1e9);
}

/** The forecast for one case file. */
function forecastCase(c: CaseFile): { hours: Array<{ hour: number; q: number[] }> } {
  const first = (c.known ?? []).find((k) => k.hour === 1);
  const p1 = Math.max(1, first && Number.isFinite(first.viewers) ? first.viewers : 1);
  const series: DaySeries[] = [];
  for (const comp of c.comparables ?? []) {
    const s = daySeries(comp);
    if (s !== null) series.push(s);
  }
  let q: Record<number, number[]>;
  if (series.length >= MIN_COMPARABLES) {
    q = smoothQuantiles(blend(knnQuantiles(p1, series, KNN_K), regLocalQuantiles(p1, series, REG_K)));
  } else {
    q = smoothQuantiles(fallbackQuantiles(p1));
  }
  const hours = HOURS.map((h) => {
    const row = q[h].map(clean);
    // Enforce non-decreasing quantiles within the row.
    for (let j = 1; j < row.length; j++) row[j] = Math.max(row[j], row[j - 1]);
    return { hour: h, q: row };
  });
  return { hours };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const out = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
