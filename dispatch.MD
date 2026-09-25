# Dispatch modelling

The **dispatch** is the control step that runs after a Team executes: the
programs' results are compared, the largest one wins, and control moves to the
winner's destination — another Team, or an Action.

It is instrumented as a whole by the `dispatch_start` / `dispatch_end` probes in
`inferenceTPG` and reported per `DispatchSize` in `latencies.json`. There is no
per-dispatch code block in the disassembly, so unlike Teams there is **no
instruction stream to featurize**. The only feature is the number of programs
compared.

---

## Quick start

```bash
# 1. Load result folders produced with the instrDispatch instrumentation
python3 main.py load <model_root> --save db_dispatch.pkl

# 2. Inspect what was measured (per-size table, Δ/size, breakpoint hint)
python3 main.py summary --db db_dispatch.pkl

# 3. Fit
python3 main.py train-dispatch --db db_dispatch.pkl --auto-breakpoint

# 4. Add --robust only if the report's RMSE/MAE ratio is above ~2, or a
#    contaminated aggregate shows up in the residual list
python3 main.py train-dispatch --db db_dispatch.pkl --auto-breakpoint --robust

# 5. Add --bootstrap 200 before quoting any coefficient
python3 main.py train-dispatch --db db_dispatch.pkl --auto-breakpoint --bootstrap 200
```

`--robust` is not free: when the data are well behaved it costs a little
efficiency. On the fixedpt set (one 333-cycle contaminant) it was essential; on
the float set it made the fit slightly *worse*. Let the RMSE/MAE ratio decide.

Existing databases must be **re-loaded** to pick up dispatch data. `refresh`
only re-parses stored disassembly; it does not re-read `latencies.json`.

---

## `train-dispatch` options

| Flag | Default | Meaning |
|---|---|---|
| `--db` | required | Database file |
| `--uarch` | all | Restrict to one uarch; otherwise every uarch with dispatch data is fitted in turn |
| `--degree` | `1` | Polynomial degree in size. `1` = straight line, `2` adds a `size²` column |
| `--breakpoint K` | none | Fit two regimes, with a level shift and a slope change at size K |
| `--auto-breakpoint` | off | Select K by grouped CV **inside the training fold** |
| `--min-segment N` | `2` | Distinct sizes required on each side of a candidate |
| `--cv-splits N` | `5` | Folds used by the selection |
| `--select-margin X` | `0` | CV MAE improvement a breakpoint must achieve to be kept |
| `--bootstrap N` | `0` | Resamples for a TPG-clustered bootstrap (CIs + K stability) |
| `--min-events N` | `100` | Ignore dispatch sizes whose **total** event count across all TPGs is below N. `0` keeps everything |
| `--min-tpgs N` | `0` | Ignore sizes measured in fewer than N TPGs |
| `--robust` | off | Fit with a Huber loss instead of squared error |
| `--unweighted` | off | Do not weight samples by their event count |
| `--test-size` | `0.20` | Fraction held out for evaluation |
| `--seed` | `42` | Split seed |

One sample = one `(TPG, uarch, dispatch size)` aggregate. Every TPG in the
database contributes, so a database built from several roots is pooled
automatically — no extra flag needed.

### Programmatic use

```python
from classes.database import Database
from analysis.dispatch import DispatchAnalyzer, DispatchRegressor

db = Database.load("db_dispatch.pkl")
samples = DispatchAnalyzer.collect_samples(db, "cv32e40x_im1_zba_zbb")
samples, dropped = DispatchAnalyzer.filter_by_events(samples, min_events=100)

model = DispatchRegressor.train(samples, breakpoint=5, robust=True)
model.print_report()
model.predict(6)              # cycles for a dispatch of size 6
model.marginal_cost_low       # cycles per extra program below the breakpoint
model.marginal_cost_high      # ... at or above it
model.step_at_breakpoint      # one-off jump at the boundary
```

---

## What the flags actually do

### `--min-events` — the threshold is per *size*, not per sample

Applied to the size's total across all TPGs. A single TPG contributes only a few
tens of dispatch events even for common sizes, so a per-sample threshold of 100
would discard the entire dataset.

A size backed by, say, 29 events over 3 TPGs is a mean of almost nothing: one
contaminated run moves it far more than any real effect. Dropped sizes are named
explicitly:

```
[train-dispatch] ignoring size 9: 29 events over 3 TPG(s) (min 100 events)
```

Note this means the model is fitted only over the retained size range, and
predictions beyond it are extrapolation.

### How K is chosen, and why it does not leak

`--auto-breakpoint` resolves K **after** the train/test split, using the
training samples only:

```
split → select K on train fold (grouped CV) → fit coefficients on train fold
      → evaluate on test fold
```

Nothing about the held-out fold influences either the coefficients or the
model's structure, so the reported test metrics need no footnote. The scan is
printed as evidence:

```
Breakpoint selection (grouped 5-fold CV, training fold only, groups = TPGs):
   candidate    CV MAE   CV RMSE  vs single
        none     5.218     7.435
         K=4     3.629     6.136     -1.588
         K=5     1.605     4.978     -3.612  <- chosen
         K=6     5.121     7.586     -0.096
```

Three deliberate choices:

- **Folds are grouped by TPG.** Samples from one TPG come from one binary and
  are correlated; letting them straddle a fold would flatter every candidate
  and reward overfitting.
- **Scored by MAE, not RMSE**, so one contaminated aggregate cannot decide the
  shape of the model.
- **A breakpoint must beat the single regime** by `--select-margin` to be kept.
  It costs two parameters and has to earn them. On genuinely linear data the
  scan returns `none` — verified against a linear control fixture.

`--bootstrap N` resamples **TPGs** (not samples — they are correlated within a
TPG, and resampling them independently gives intervals that are far too narrow)
and reports how often each K is re-selected. On a real step this reads
`K=5: 100% (unanimous)`; on linear data it scatters, e.g. `none: 68%, K=6: 25%`.
That distribution is the honest answer to "is this breakpoint real".

### `--breakpoint K` — fitting a step

Adds two columns to the design matrix:

```
step_geK        = 1 if size >= K else 0        (level shift)
extra_slope_geK = max(0, size - (K-1))         (hinge, = 1 at size K)
```

Still linear in the coefficients, so it is still plain least squares. The report
translates them into readable quantities: marginal cost below K, marginal cost at
or above K, and the one-off step.

Guarded by `--min-segment` (default 2). Two is the *identifiability* minimum:
each regime has a bias and a slope, and estimating a slope needs two distinct x
values. But a segment with exactly two sizes has zero residual degrees of
freedom — its line passes exactly through both means, so its linearity is
untested. The report warns when this happens; `--min-segment 3` requires the
stronger condition.

**Multiple breakpoints are not supported, deliberately.** With N distinct sizes
and N−1 breakpoints the model has one parameter per size and becomes identical
to the saturated baseline: a perfect fit with no interpretation and no ability
to extrapolate. If a second breakpoint is ever warranted, the guards are
`--min-segment 3` (so each regime's linearity is actually tested), a
`--select-margin` it has to clear, and a physical story in the disassembly — a
breakpoint should correspond to a discrete codegen event, and you would expect
zero or one of those, not five.

**A polynomial cannot represent a step.** If the marginal cost jumps at one
particular size, `--degree 2` will not help; the report will say so:

```
Hint: the marginal cost changes sharply around size 5. A polynomial cannot fit
a step — try `--auto-breakpoint`, which selects it on the training fold so the
test metrics stay honest.
```

The hint deliberately does **not** tell you to pass `--breakpoint 5`: it is
computed from all samples, so acting on it by hand would let the held-out fold
influence the model's structure — the leak `--auto-breakpoint` exists to avoid.

### `--robust` — Huber loss

Least squares minimises `Σe²`, so a residual of 200 cycles weighs 10 000× one of
2 — a single contaminated aggregate can set the whole fit. Huber is quadratic
near zero and **linear** beyond `ε·scale` (ε = 1.35), so an extreme point's
influence saturates.

It is not outlier deletion. The point stays in the dataset, still contributes to
the reported RMSE, and is still listed in the residual table — it just stops
dictating the coefficients. Expect train metrics to stay ugly under `--robust`
while the coefficients become sane; that is correct behaviour.

Implementation: swaps `LinearRegression` for `HuberRegressor(epsilon=1.35,
alpha=0.0)`. `alpha=0` disables the default L2 penalty so coefficients stay
comparable. Nothing else changes.

### `--unweighted`

By default each sample is weighted by its event count. Each label is already a
mean over `Count` events, so its variance is ∝ σ²/Count and the counts vary
widely (10 to ~2000). Weighted least squares is the statistically correct fit
here, not a heuristic.

---

## Reading the report

**MAE** — mean absolute error, in cycles. Your typical miss.

**RMSE** — root mean squared error. Same units, but squaring means one big miss
counts far more than several small ones. Always ≥ MAE.

**R²** — fraction of the label variance explained, baseline "always predict the
mean". Unitless, but inflated when labels span a wide range — which they do here
(22 → 180 cycles), so a high R² alone means little.

**RMSE/MAE** — the diagnostic. ~1.2–1.3 for well-behaved errors. Above 2 means a
handful of samples dominate the squared loss; the report flags this.

**Saturated baseline** — a reference model that predicts each size by that size's
own weighted mean (computed on the training fold). A lookup table with one free
parameter per size — hence "saturated". No model whose only input is dispatch
size can beat it, so it is *the ceiling*. The gap line is your model's RMSE minus
the baseline's:

- gap ≫ 0 → wrong functional form; a better shape exists
- gap ≈ 0 → the form is as good as it gets; the remaining error is variation
  *within* each size, invisible to a size-only model

**Noise floor** — RMSE that measurement error alone would produce, from the
per-sample standard errors (`stddev/√count`). If test RMSE sits near it, the
model is as accurate as the data permits.

**Negative bias warning** — a negative fixed overhead is physically impossible (a
dispatch of size 0 cannot cost less than nothing). It means the line is tilting
to straddle two regimes; use `--breakpoint`.

**Per-size breakdown** — a `*` marks sizes resting on fewer than 5 TPGs or 100
dispatch events. A residual on a starred row is thin evidence whatever the fit
statistics say.

**Residual list** — samples whose residual exceeds 3× the mean absolute residual,
with the TPG name and event count so they can be investigated. Shown for both
estimators; the wording changes depending on whether the fit resisted them.

---

## Results

### fixedpt — `cv32e40x_im1_zba_zbb` and `cv32e40x_im2_zba_zbb`

`--auto-breakpoint --robust`, sizes below 100 events dropped:

| | im1 (sizes 2–8, n=156) | im2 (sizes 2–6, n=95) |
|---|---|---|
| fixed overhead | 7.36 | 7.57 |
| cycles/program, size < 5 | 7.43 | 7.33 |
| step at size 5 | +21.33 | +20.67 |
| cycles/program, size ≥ 5 | 9.64 | 10.37 |
| test RMSE | 2.36 | 1.77 |
| gap to saturated | −0.010 | +0.022 |

Two uarchs, different TPG sets (31 vs 21), agreeing to within a few percent.
Since im1 and im2 differ in a multiplier configuration the dispatch code does not
use, this is a physical consistency check that passed.

### float — `cv32e40px_fpu`

31 TPGs across three families (logexp, logexp_trig, trig), 167 samples,
`--breakpoint 4`, least squares:

```
cycles ≈ 9.55 + 12.67·size + 14.05·[size ≥ 4] + 0.44·(size-3)+
```

| | value |
|---|---|
| fixed overhead | 9.55 |
| cycles/program, size < 4 | 12.67 |
| step at size 4 | +14.49 |
| cycles/program, size ≥ 4 | 13.11 |
| test RMSE | 4.20 |
| gap to saturated | −0.21 |

Note the break sits at **4** here, not 5, and — unlike fixedpt — the slope does
not change across it: `extra_slope` is +0.44 under least squares and −0.38 under
Huber, flipping sign, which is what a coefficient does when its true value is
zero. The float dispatch is **a straight line with a pure one-off jump**.

The gap to the saturated baseline is ≈0 on every uarch: **the functional form is
finished**. No further basis change can improve a size-only model.

---

## Why the step exists — confirmed from the disassembly

The step is a **compiler artefact, not a hardware property**: GCC fully unrolls
`bestProgram` for small trip counts and emits a rolled loop beyond a threshold.

`T3_end` (size 3, float build) — fully unrolled, no backward branch anywhere:

```
db0: flw   fa5,28(sp)      scores[0]
dbc: flw   fa4,32(sp)      scores[1]     immediate offset
dc8: fle.s a5,fa5,fa4      compare
dd0: flw   fa4,36(sp)      scores[2]     immediate offset
dd8: li    a3,0            winning index as a CONSTANT
e08: li    a3,2
```

`T4_end` (size 4, same build) — rolled loop:

```
e90: addi a4,sp,92         pointer setup   \
e94: li   a1,0             best index       |
e98: li   a5,1             i = 1            > paid once, per dispatch
e9c: li   a2,4             n = 4            |
ea0: j    ec0              enter loop      /
...
eb4: addi a5,a5,1          i++             \
eb8: addi a4,a4,4          ptr += 4         > per program
ebc: beq  a5,a2,ee0        loop exit       /
```

So GCC's complete-unroll threshold sits between trip count 2 and 3. Costing the
one-off part: 5 setup instructions (~8 cycles with the taken `j`) plus the
loop-exit `beq` (~3) gives **11–14 cycles**, against a fitted step of **+14.49**
(LS) and **+12.98** (Huber). A direct hit.

This also explains the two things the fit reported that looked odd:

- **Why `extra_slope` ≈ 0 on float.** The unrolled chain and the loop body cost
  about the same per program — the loop pays `addi`/`addi`/`beq` for
  bookkeeping, the unrolled chain pays extra branch penalties in its longer
  cascade. They roughly cancel.
- **Why fixedpt breaks at 5 and float at 4.** The integer comparison body is
  smaller, so the unroll heuristic tolerates one more iteration before rolling.

The 3→4 delta being almost exactly double the others (27.25 vs ~13.3) is a
coincidence: the loop setup cost happens to land near the per-program cost. Not
structure — do not read significance into it.

The model is therefore mechanistically grounded rather than curve-fitted:

```
cycles ≈ 9.55 + 12.67·size + 14.05·[size ≥ 4]
          ↑        ↑              ↑
      probe +  per-program    loop prologue + exit
    jump table  comparison     (rolled form only)
```

**Consequence for the writeup:** the break location depends on optimisation
flags, not on the core. It will move if the build changes, and should not be
presented as a property of `cv32e40x`. Compiling with `-fno-unroll-loops` (or
forcing full unrolling) would remove the discontinuity and give a single-regime
model, if a clean linear cost is worth more to you than the cycles.

### An earlier hypothesis that was wrong

A previous version of this document attributed the step to **register
spilling** — supposedly the compiler allocating a stack frame once the program
results no longer fit in caller-saved registers. The disassembly disproves it:
the `scores[]` array is on the stack in every version, and the `flw fs0..fs4`
restores visible at size 10 are callee-saved registers holding `in1..in4`, whose
count does not depend on dispatch size. Kept here as a record of what was ruled
out.

---

## What the remaining error is

At well-measured sizes the spread across TPGs is ~10–12 cycles (float size 4:
71..84), an order of magnitude above the 0.2–0.6 noise floor. Real and
systematic, but invisible to a size-only model.

The likely source is in the loop body. The update path

```
fmv.s fa4,fa5    /  mv a1,a5  /  j
```

executes **only when a new running maximum is found**, so the dispatch cost
depends on the *ordering* of the scores. Each TPG draws that ordering once at
build time and keeps it across all its runs, so each TPG has its own stable
offset. Across TPGs the mean number of updates is ≈ ln(n) — absorbed into the
fitted slope, contributing no scatter — but the variance of that draw is also
≈ ln(n), giving a standard deviation of roughly one update, i.e. a few cycles,
with a full range across 20–30 TPGs of about 10. That matches what is observed.

Capturing it would need a per-dispatch feature (the number of max-updates, or a
proxy for the score ordering), not a better regression. The orderings are also
not uniformly random — training may correlate a program's position with its
typical score — so the real spread can differ from the uniform-random estimate.

---

## Known limitation

With `--degree > 1` **and** a breakpoint, the "marginal cost per extra program"
block is wrong: it reports `w1` and `w1 + w_extra`, ignoring the quadratic term's
contribution to the derivative. It is correct for `--degree 1`, which is what the
data supports anyway — the degree-2 fits produced large cancelling terms
(`+12.99·size − 1.00·size²`), the classic sign of an over-specified basis, and
fit worse on held-out data.

---

## Reporting checklist

Before quoting a dispatch model:

1. **No duplicate TPGs** in the database (see Data hygiene).
2. **`--auto-breakpoint`**, not a hand-picked `--breakpoint K` read off the
   summary — otherwise the test metrics carry a selection-leak caveat.
3. **`--bootstrap 200`** for coefficient intervals and K stability. Quote the
   intervals, not just the point estimates.
4. **Check the gap to saturated.** ≈0 is the claim that the functional form is
   complete; report it.
5. **Check `--min-segment` warnings.** A segment with two distinct sizes has an
   untested slope — say so rather than quoting it as measured.
6. **State that the break is a codegen threshold**, with the optimisation flags
   used, so it is not mistaken for a property of the core.

---

## Implementation map

| File | Role |
|---|---|
| `analysis/disassembler.py` | `DispatchLatency`; parses the `Dispatches` array from `instrDispatch_instrTeams_instrTPG`. Optional — `has_dispatch_data` distinguishes absent from present-but-empty |
| `classes/dispatchMeasurements.py` | `DispatchMeasurement` — mirrors `TeamMeasurement`, adds `dispatch_size` and `sem` |
| `classes/tpg.py` | `TPG.dispatch_latencies[uarch][size]`, alongside `class_latencies`. Keyed by uarch because dispatch cost is a timing property. `__setstate__` defaults it for older pickles |
| `classes/loader.py` | Stores the measurements; also finds the disassembly by candidate list + glob, since the dispatch build renames it to `disassembly_tpg_inference_instrDispatch_instrTeams_instrTPG.txt` |
| `analysis/dispatch.py` | `DispatchSample`, `DispatchAnalyzer` (collection, `filter_by_events`, per-size stats, `select_breakpoint` via grouped CV), `BreakpointSelection`, `BootstrapInterval`, `DispatchModel`, `DispatchRegressor` (fit, train-fold selection, TPG-clustered bootstrap) |
| `main.py` | `train-dispatch` subcommand |
| `classes/database.py` | `print_summary` gains a dispatch section |