"""
analysis/dispatch.py — modelling the TPG dispatch from its size.

The dispatch is the control step that runs *after* a Team has executed: the
programs' results are compared, the largest one wins, and control moves to the
winner's destination (another Team, or an Action).  It is instrumented as a
whole in ``inferenceTPG`` and reported per ``DispatchSize`` in
``latencies.json``, so — unlike Teams — there is no instruction stream to
extract features from.

The model is therefore a small, explicit one::

    cycles(dispatch) ≈ bias + w1·size [+ w2·size² …] [+ piecewise terms]

where ``bias`` is the fixed dispatch overhead (probe, bookkeeping, the final
jump) and ``w1`` is the marginal cost of comparing one more program.  Samples
are pooled across every TPG and every loaded folder in the Database, exactly
like the team-level pipeline: one sample = one (TPG, uarch, dispatch size)
aggregate.

Three details matter for getting sensible numbers out of it:

*   Each sample is already an average over ``Count`` dispatch events, so its
    precision varies a lot (Count ranges from ~10 to ~2000 in practice).
    Fitting is weighted by ``Count`` by default so that well-measured points
    count for more — ordinary weighted least squares, not a heuristic.
*   The relation need not be a single straight line.  Measurements show the
    marginal cost changing abruptly at a given size (a codegen change:
    register spills, or an unrolled comparison chain becoming a loop).  A
    polynomial cannot represent a step, so ``breakpoint`` adds a level shift
    and a slope change at a chosen size — still linear in the coefficients,
    still plain least squares.
*   Size alone can only ever explain so much.  The *saturated* baseline
    (predict each size's own weighted mean) is the best any size-only model
    can do, so comparing against it separates "wrong functional form" from
    "dispatch size is not sufficient".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from analysis.regression import RegressionMetrics, TrainTestSplit

if TYPE_CHECKING:
    from classes.database import Database


# A per-size aggregate resting on fewer samples/events than these is flagged
# in the report: its mean is thin evidence, however confident the fit looks.
_SPARSE_MIN_SAMPLES = 5
_SPARSE_MIN_EVENTS  = 100


# --------------------------------------------------------------------------- #
# Design matrix
# --------------------------------------------------------------------------- #

def _feature_names(degree: int, breakpoint: int | None = None) -> list[str]:
    """
    Basis feature names, in coefficient order.

    Polynomial terms first, then — if *breakpoint* is set — the two piecewise
    terms that let the fit change level and slope at that size.
    """
    names = ["dispatch_size"]
    for d in range(2, degree + 1):
        names.append(f"dispatch_size^{d}")
    if breakpoint is not None:
        names.append(f"step_ge{breakpoint}")
        names.append(f"extra_slope_ge{breakpoint}")
    return names


def _design_row(size: float, degree: int,
                breakpoint: int | None = None) -> list[float]:
    """
    One row of the design matrix for a dispatch of *size*.

    With a breakpoint *b* the two extra columns are

        step_ge{b}         = 1 if size >= b else 0       (level shift)
        extra_slope_ge{b}  = max(0, size - (b - 1))      (hinge, 1 at size b)

    so the fitted marginal cost is ``w1`` below *b* and ``w1 + w_extra`` at or
    above it, with a one-off jump of ``w_step + w_extra`` at the boundary.
    """
    row = [float(size) ** d for d in range(1, degree + 1)]
    if breakpoint is not None:
        row.append(1.0 if size >= breakpoint else 0.0)
        row.append(max(0.0, float(size) - (breakpoint - 1)))
    return row


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #

@dataclass
class DispatchSample:
    """
    One training point: the measured cost of dispatches of one size, in one
    TPG, on one uarch.

    Attributes
    ----------
    tpg_name:        Display name of the owning TPG.
    tpg_source_path: Absolute seed-dir path (unique TPG identity).
    uarch_name:      Micro-architecture the measurement was taken on.
    dispatch_size:   Number of programs compared — the only raw feature.
    latency:         Measured AvgCyclesPerDispatch (the label).
    nb_measurements: Number of dispatch events averaged (the fit weight).
    stddev:          Spread across those events.
    """
    tpg_name: str
    tpg_source_path: str
    uarch_name: str
    dispatch_size: int
    latency: float
    nb_measurements: int
    stddev: float

    @property
    def sem(self) -> float:
        """Standard error of this sample's mean (label precision)."""
        if self.nb_measurements <= 0:
            return 0.0
        return self.stddev / (self.nb_measurements ** 0.5)


@dataclass
class DispatchSizeStat:
    """Aggregate of every sample sharing one dispatch size, for reporting."""
    dispatch_size: int
    n_samples: int
    n_events: int
    mean_latency: float
    min_latency: float
    max_latency: float
    predicted: float = 0.0

    @property
    def residual(self) -> float:
        return self.mean_latency - self.predicted

    @property
    def spread(self) -> float:
        """max − min across the TPGs contributing to this size."""
        return self.max_latency - self.min_latency

    @property
    def is_sparse(self) -> bool:
        """True when this row rests on too little data to trust its mean."""
        return (self.n_samples < _SPARSE_MIN_SAMPLES
                or self.n_events < _SPARSE_MIN_EVENTS)


@dataclass
class BreakpointCandidate:
    """Cross-validated score of one candidate breakpoint (None = single regime)."""
    breakpoint: int | None
    cv_mae: float
    cv_rmse: float
    n_folds: int
    chosen: bool = False


@dataclass
class BreakpointSelection:
    """
    Outcome of choosing a breakpoint, with the evidence behind it.

    The scan runs entirely inside the training fold, using grouped
    cross-validation with TPGs as groups: no held-out sample influences the
    choice, and samples from one TPG never straddle a CV fold (they come from
    one binary and are correlated, so splitting them would flatter every
    candidate and reward overfitting).

    Attributes
    ----------
    candidates:   Every candidate scored, single-regime first.
    chosen:       Selected breakpoint, or None if no candidate beat the
                  single-regime baseline by ``margin``.
    baseline_mae: CV MAE of the single-regime model.
    margin:       Improvement in CV MAE a breakpoint had to achieve to be kept.
    min_segment:  Distinct sizes required on each side of a candidate.
    """
    candidates: list[BreakpointCandidate] = field(default_factory=list)
    chosen: int | None = None
    baseline_mae: float = 0.0
    margin: float = 0.0
    min_segment: int = 2

    def print_table(self, indent: str = "  ") -> None:
        if not self.candidates:
            return
        print(f"\n{indent}Breakpoint selection "
              f"(grouped {self.candidates[0].n_folds}-fold CV, training fold "
              f"only, groups = TPGs):")
        print(f"{indent}  {'candidate':>10} {'CV MAE':>9} {'CV RMSE':>9} "
              f"{'vs single':>10}")
        for c in self.candidates:
            label = "none" if c.breakpoint is None else f"K={c.breakpoint}"
            delta = ("" if c.breakpoint is None
                     else f"{c.cv_mae - self.baseline_mae:+.3f}")
            mark = "  <- chosen" if c.chosen else ""
            print(f"{indent}  {label:>10} {c.cv_mae:>9.3f} {c.cv_rmse:>9.3f} "
                  f"{delta:>10}{mark}")
        if self.chosen is None:
            print(f"{indent}  no candidate improved on the single regime by "
                  f"more than {self.margin:.3f} MAE — fitting one regime")


@dataclass
class BootstrapInterval:
    """Percentile confidence interval for one quantity from a cluster bootstrap."""
    name: str
    point: float
    low: float
    high: float

    def __repr__(self) -> str:
        return f"{self.name}={self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

@dataclass
class DispatchModel:
    """
    A fitted dispatch model: ``cycles ≈ bias + Σ w_k · basis_k(size)``.

    Attributes
    ----------
    weights:        Coefficient per basis feature.
    bias:           Intercept — the size-independent dispatch overhead.
    degree:         Polynomial degree in size (1 = affine).
    breakpoint:     Size at which a level shift and slope change are allowed,
                    or None for a single regime.
    weighted:       Whether samples were weighted by their event count.
    feature_names:  Basis feature names, in coefficient order.
    split:          Train/test split information.
    train_metrics:  MAE / RMSE / R² on the training fold.
    test_metrics:   MAE / RMSE / R² on the held-out fold.
    saturated_train_metrics / saturated_test_metrics:
                    Same metrics for the saturated baseline — predicting each
                    size's own weighted mean (fitted on the training fold).
                    This is the ceiling for any model using size alone.
    size_stats:     Per-size aggregates with the model's prediction.
    noise_floor:    RMSE that measurement noise alone would produce
                    (root-mean-square of the per-sample standard errors).
    """
    weights:       dict[str, float]
    bias:          float
    degree:        int = 1
    breakpoint:    int | None = None
    weighted:      bool = True
    feature_names: list[str] = field(default_factory=list)
    split:         TrainTestSplit | None = None
    train_metrics: RegressionMetrics | None = None
    test_metrics:  RegressionMetrics | None = None
    saturated_train_metrics: RegressionMetrics | None = None
    saturated_test_metrics:  RegressionMetrics | None = None
    size_stats:    list[DispatchSizeStat] = field(default_factory=list)
    noise_floor:   float = 0.0
    estimator:     str = "least squares"
    flagged_outliers: list["DispatchSample"] = field(default_factory=list)
    selection:     "BreakpointSelection | None" = None
    bootstrap:     list["BootstrapInterval"] = field(default_factory=list)
    bootstrap_breakpoints: dict = field(default_factory=dict)
    thin_segments: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(self, dispatch_size: float) -> float:
        """Predicted cycle cost of a dispatch comparing *dispatch_size* programs."""
        total = self.bias
        names = _feature_names(self.degree, self.breakpoint)
        row = _design_row(dispatch_size, self.degree, self.breakpoint)
        for name, value in zip(names, row):
            total += self.weights.get(name, 0.0) * value
        return total

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #

    @property
    def marginal_cost_low(self) -> float:
        """Cycles added by one more program below the breakpoint."""
        return self.weights.get("dispatch_size", 0.0)

    @property
    def marginal_cost_high(self) -> float:
        """Cycles added by one more program at/above the breakpoint."""
        if self.breakpoint is None:
            return self.marginal_cost_low
        return self.marginal_cost_low + self.weights.get(
            f"extra_slope_ge{self.breakpoint}", 0.0)

    @property
    def step_at_breakpoint(self) -> float:
        """One-off jump in cycles when crossing into the upper regime."""
        if self.breakpoint is None:
            return 0.0
        return self.predict(self.breakpoint) - (
            self.predict(self.breakpoint - 1) + self.marginal_cost_low)

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #

    def formula(self) -> str:
        """Human-readable formula."""
        parts = [f"{self.bias:.2f}"]
        for name in _feature_names(self.degree, self.breakpoint):
            w = self.weights.get(name, 0.0)
            sign = "+" if w >= 0 else "-"
            if name == "dispatch_size":
                term = "size"
            elif name.startswith("dispatch_size^"):
                term = f"size^{name.split('^')[1]}"
            elif name.startswith("step_ge"):
                term = f"[size>={self.breakpoint}]"
            else:
                term = f"(size-{self.breakpoint - 1})+"
            parts.append(f"{sign} {abs(w):.2f}·{term}")
        return "cycles ≈ " + " ".join(parts)

    def __repr__(self) -> str:
        bp = f", breakpoint={self.breakpoint}" if self.breakpoint else ""
        return (f"DispatchModel({self.formula()}, degree={self.degree}{bp}, "
                f"weighted={self.weighted})")

    def print_report(self) -> None:
        """Print a human-readable dispatch-regression report."""
        bar = "─" * 78
        print(f"\n{bar}")
        print("  DISPATCH REGRESSION REPORT")
        print(bar)
        if self.split:
            s = self.split
            print(f"  Dataset split (seed={s.random_state}): "
                  f"{s.n_train} train  /  {s.n_test} test  (total {s.n_total})")
        basis = f"degree {self.degree}"
        if self.breakpoint is not None:
            basis += f", piecewise at size {self.breakpoint}"
        print(f"  Basis: {basis}"
              f"   weighting: {'by event count' if self.weighted else 'none'}"
              f"   estimator: {self.estimator}")
        print(f"  Model: {self.formula()}")
        print(f"  Fixed overhead (bias): {self.bias:.3f} cycles"
              + ("   <-- negative: physically impossible, the fit is straddling "
                 "two regimes" if self.bias < 0 else ""))
        for name in _feature_names(self.degree, self.breakpoint):
            print(f"  Coefficient {name:<24} {self.weights.get(name, 0.0):+.4f}")

        if self.breakpoint is not None:
            print(f"\n  Marginal cost per extra program:")
            print(f"    size <  {self.breakpoint}:  "
                  f"{self.marginal_cost_low:+.2f} cycles")
            print(f"    size >= {self.breakpoint}:  "
                  f"{self.marginal_cost_high:+.2f} cycles")
            print(f"    one-off step at size {self.breakpoint}: "
                  f"{self.step_at_breakpoint:+.2f} cycles")

        if self.thin_segments:
            for msg in self.thin_segments:
                print(f"  Note: {msg}")

        if self.selection is not None:
            self.selection.print_table(indent="  ")

        if self.bootstrap:
            n_boot = sum(self.bootstrap_breakpoints.values())
            print(f"\n  Cluster bootstrap over TPGs ({n_boot} resamples, "
                  f"95% percentile intervals):")
            for ci in self.bootstrap:
                print(f"    {ci.name:<26} {ci.point:>9.3f}  "
                      f"[{ci.low:>8.3f}, {ci.high:>8.3f}]")
            if self.selection is not None and self.bootstrap_breakpoints:
                picks = ", ".join(
                    f"{'none' if k is None else f'K={k}'}: {v / n_boot:.0%}"
                    for k, v in sorted(self.bootstrap_breakpoints.items(),
                                       key=lambda kv: -kv[1]))
                print(f"    breakpoint re-selected as  {picks}"
                      + ("   (unanimous)"
                         if len(self.bootstrap_breakpoints) == 1 else ""))

        if self.train_metrics:
            print(f"\n  Train metrics:  {self.train_metrics}")
        if self.test_metrics:
            print(f"  Test  metrics:  {self.test_metrics}")
            ratio = (self.test_metrics.rmse / self.test_metrics.mae
                     if self.test_metrics.mae else 0.0)
            note = ("  <-- >2 means a few outliers dominate the squared error"
                    if ratio > 2.0 else "")
            print(f"  RMSE/MAE     :  {ratio:.2f}{note}")

        if self.saturated_test_metrics or self.saturated_train_metrics:
            print(f"\n  Saturated baseline (predict each size's own mean — the "
                  f"ceiling for any size-only model):")
            if self.saturated_train_metrics:
                print(f"    train: {self.saturated_train_metrics}")
            if self.saturated_test_metrics:
                print(f"    test : {self.saturated_test_metrics}")
                if self.test_metrics:
                    gap = self.test_metrics.rmse - self.saturated_test_metrics.rmse
                    print(f"    gap to this model: RMSE {gap:+.3f}  "
                          f"(≈0 ⇒ the functional form is right and the "
                          f"remaining error is not explainable by size)")
        if self.noise_floor:
            print(f"\n  Noise floor  :  RMSE={self.noise_floor:.3f}  "
                  f"(measurement error alone)")

        if self.flagged_outliers:
            resisted = (" (the robust fit resisted them; they barely moved "
                        "the coefficients)" if self.estimator == "huber"
                        else " (these are pulling the coefficients — consider "
                             "--robust)")
            print(f"\n  Samples with residuals over 3x the typical error "
                  f"({len(self.flagged_outliers)}){resisted}:")
            worst = sorted(self.flagged_outliers,
                           key=lambda s: -abs(s.latency - self.predict(s.dispatch_size)))
            for smp in worst[:8]:
                pred = self.predict(smp.dispatch_size)
                print(f"    size {smp.dispatch_size:>2}  measured "
                      f"{smp.latency:>8.1f}  predicted {pred:>8.1f}  "
                      f"({smp.latency - pred:+.1f})  "
                      f"{smp.nb_measurements:>5} events  TPG {smp.tpg_name}")
            if len(worst) > 8:
                print(f"    … and {len(worst) - 8} more")

        if self.size_stats:
            print(f"\n  Per-size breakdown:")
            header = (f"    {'size':>4} {'n':>4} {'events':>7} "
                      f"{'measured':>10} {'predicted':>10} {'residual':>10} "
                      f"{'min..max':>17}  ")
            print(header)
            print("    " + "-" * (len(header) - 4))
            any_sparse = False
            for st in self.size_stats:
                span = f"{st.min_latency:.0f}..{st.max_latency:.0f}"
                mark = "*" if st.is_sparse else " "
                any_sparse = any_sparse or st.is_sparse
                print(f"    {st.dispatch_size:>4} {st.n_samples:>4} "
                      f"{st.n_events:>7} {st.mean_latency:>10.2f} "
                      f"{st.predicted:>10.2f} {st.residual:>+10.2f} "
                      f"{span:>17} {mark}")
            if any_sparse:
                print(f"    * thin evidence: fewer than {_SPARSE_MIN_SAMPLES} "
                      f"TPGs or {_SPARSE_MIN_EVENTS} dispatch events — treat "
                      f"the mean and its residual with caution")
        print(bar)


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #

class DispatchAnalyzer:
    """
    Collects dispatch measurements out of a Database.  All methods are
    static — this is a namespace.
    """

    @staticmethod
    def collect_samples(
        db: "Database",
        uarch_name: str | None = None,
    ) -> list[DispatchSample]:
        """
        Gather one DispatchSample per (TPG, uarch, dispatch size) in *db*.

        Every loaded folder contributes its TPGs, so a database built from
        several roots trains on all of them at once.

        Args:
            db:         Database to read.
            uarch_name: Restrict to one uarch; all uarchs when None.

        Returns:
            Samples ordered by TPG, then uarch, then dispatch size.
        """
        samples: list[DispatchSample] = []
        for tpg in db.tpgs.values():
            names = ([uarch_name] if uarch_name is not None
                     else sorted(tpg.dispatch_latencies))
            for name in names:
                dispatches = tpg.get_dispatches(name)
                for size in sorted(dispatches):
                    dm = dispatches[size]
                    samples.append(
                        DispatchSample(
                            tpg_name=tpg.name,
                            tpg_source_path=tpg.source_path,
                            uarch_name=name,
                            dispatch_size=size,
                            latency=dm.latency,
                            nb_measurements=dm.nb_measurements,
                            stddev=dm.stddev,
                        )
                    )
        return samples

    @staticmethod
    def filter_by_events(
        samples: list[DispatchSample],
        min_events: int = _SPARSE_MIN_EVENTS,
        min_tpgs: int = 0,
    ) -> tuple[list[DispatchSample], list[DispatchSizeStat]]:
        """
        Drop every sample whose *dispatch size* rests on too little evidence.

        The threshold is applied to the size's **total** across all TPGs, not
        to individual samples: one TPG contributes only a few tens of dispatch
        events even for common sizes, so a per-sample threshold would discard
        everything.  A size backed by, say, 29 events spread over 3 TPGs is a
        mean of almost nothing, and a single contaminated run there moves it
        far more than any real effect — such rows distort a fit while
        contributing nothing it can learn from.

        Args:
            samples:    Samples to filter.
            min_events: Minimum total dispatch events for a size to be kept.
            min_tpgs:   Minimum number of contributing TPGs for a size.

        Returns:
            ``(kept_samples, dropped_size_stats)`` — the stats of the removed
            sizes, so the caller can report exactly what was excluded.
        """
        stats = DispatchAnalyzer.size_stats(samples)
        dropped = [st for st in stats
                   if st.n_events < min_events or st.n_samples < min_tpgs]
        dropped_sizes = {st.dispatch_size for st in dropped}
        kept = [s for s in samples if s.dispatch_size not in dropped_sizes]
        return kept, dropped

    @staticmethod
    def uarchs_with_dispatch_data(db: "Database") -> list[str]:
        """Uarch names for which at least one TPG has dispatch measurements."""
        names: set[str] = set()
        for tpg in db.tpgs.values():
            names.update(n for n, d in tpg.dispatch_latencies.items() if d)
        return sorted(names)

    @staticmethod
    def size_stats(samples: list[DispatchSample]) -> list[DispatchSizeStat]:
        """Aggregate *samples* by dispatch size."""
        by_size: dict[int, list[DispatchSample]] = {}
        for s in samples:
            by_size.setdefault(s.dispatch_size, []).append(s)

        stats: list[DispatchSizeStat] = []
        for size in sorted(by_size):
            group = by_size[size]
            lats = [s.latency for s in group]
            stats.append(
                DispatchSizeStat(
                    dispatch_size=size,
                    n_samples=len(group),
                    n_events=sum(s.nb_measurements for s in group),
                    mean_latency=sum(lats) / len(lats),
                    min_latency=min(lats),
                    max_latency=max(lats),
                )
            )
        return stats

    @staticmethod
    def _deltas(stats: list[DispatchSizeStat]) -> list[tuple[int, float]]:
        """``(size, Δ mean latency per unit of size)`` for consecutive stats."""
        out: list[tuple[int, float]] = []
        for prev, cur in zip(stats, stats[1:]):
            gap = cur.dispatch_size - prev.dispatch_size
            if gap > 0:
                out.append((cur.dispatch_size,
                            (cur.mean_latency - prev.mean_latency) / gap))
        return out

    @staticmethod
    def marginal_costs(samples: list[DispatchSample]) -> list[tuple[int, float]]:
        """
        Step-to-step differences of the per-size means: ``(size, Δ from the
        previous size, per unit of size)``.

        A constant sequence means one linear regime; a jump locates a
        breakpoint candidate.
        """
        return DispatchAnalyzer._deltas(DispatchAnalyzer.size_stats(samples))

    @staticmethod
    def suggest_breakpoint(samples: list[DispatchSample]) -> int | None:
        """
        Size at which the marginal cost changes most sharply, or None if the
        data look like a single regime.

        Sparse sizes are excluded first: a mean resting on one or two TPGs
        moves far more from a single contaminated measurement than from any
        real change of regime, and would otherwise capture the suggestion.

        A hint for ``--breakpoint``, not a decision: it reports where the
        largest jump in per-program cost occurs, provided that jump is at
        least half again the median marginal cost.
        """
        stats = [st for st in DispatchAnalyzer.size_stats(samples)
                 if not st.is_sparse]
        deltas = DispatchAnalyzer._deltas(stats)
        if len(deltas) < 3:
            return None
        values = sorted(d for _s, d in deltas)
        median = values[len(values) // 2]
        size, worst = max(deltas, key=lambda sd: sd[1])
        if median > 0 and worst > 1.5 * median:
            return size
        return None

    @staticmethod
    def select_breakpoint(
        samples:      list[DispatchSample],
        degree:       int = 1,
        weighted:     bool = True,
        robust:       bool = False,
        min_segment:  int = 2,
        n_splits:     int = 5,
        margin:       float = 0.0,
    ) -> "BreakpointSelection":
        """
        Choose a breakpoint by grouped cross-validation *within the samples
        given* — never on a held-out fold.

        Every candidate size with at least ``min_segment`` distinct sizes on
        each side is scored, along with the single-regime baseline, by
        out-of-fold weighted MAE.  Folds are grouped by TPG so that samples
        from one binary never appear in both the fit and the score of a fold;
        they are correlated, and splitting them would make every candidate
        look better than it is.

        MAE is the score rather than RMSE because a single contaminated
        aggregate should not decide the shape of the model.

        A candidate is chosen only if it improves on the single-regime
        baseline by more than ``margin`` — a breakpoint costs two parameters
        and has to earn them.

        Args:
            samples:     Samples to select on (typically the training fold).
            degree:      Polynomial degree of the base model.
            weighted:    Weight samples by their event count.
            robust:      Score using the same estimator that will be fitted.
            min_segment: Distinct sizes required on each side of a candidate.
            n_splits:    Cross-validation folds (capped by the number of TPGs).
            margin:      MAE improvement a breakpoint must achieve.

        Returns:
            A :class:`BreakpointSelection` holding the scan and the choice.
        """
        import numpy as np
        from sklearn.model_selection import GroupKFold

        sizes = sorted({s.dispatch_size for s in samples})
        groups = np.array([s.tpg_source_path for s in samples])
        n_groups = len(set(groups))
        n_splits = max(2, min(n_splits, n_groups))

        y = np.array([s.latency for s in samples], dtype=float)
        w = np.array([max(s.nb_measurements, 1) for s in samples], dtype=float)

        candidates: list[int | None] = [None] + _candidate_breakpoints(
            sizes, min_segment)

        results: list[BreakpointCandidate] = []
        for bp in candidates:
            X = np.array([_design_row(s.dispatch_size, degree, bp)
                          for s in samples])
            errs: list[float] = []
            wts:  list[float] = []
            splitter = GroupKFold(n_splits=n_splits)
            for tr, va in splitter.split(X, y, groups=groups):
                if len(tr) < X.shape[1] + 1:
                    continue
                try:
                    model = _fit_estimator(X[tr], y[tr], w[tr], robust, weighted)
                    pred = model.predict(X[va])
                except Exception:
                    continue
                errs.extend(np.abs(y[va] - pred))
                wts.extend(w[va])
            if not errs:
                continue
            e = np.array(errs)
            ww = np.array(wts)
            results.append(
                BreakpointCandidate(
                    breakpoint=bp,
                    cv_mae=float(np.average(e, weights=ww)),
                    cv_rmse=float(np.sqrt(np.average(e ** 2, weights=ww))),
                    n_folds=n_splits,
                )
            )

        selection = BreakpointSelection(candidates=results, margin=margin,
                                        min_segment=min_segment)
        baseline = next((r for r in results if r.breakpoint is None), None)
        if baseline is None:
            return selection
        selection.baseline_mae = baseline.cv_mae

        with_bp = [r for r in results if r.breakpoint is not None]
        if with_bp:
            best = min(with_bp, key=lambda r: r.cv_mae)
            if best.cv_mae < baseline.cv_mae - margin:
                best.chosen = True
                selection.chosen = best.breakpoint
                return selection
        baseline.chosen = True
        return selection

    # ------------------------------------------------------------------ #
    # Summary printing
    # ------------------------------------------------------------------ #

    @staticmethod
    def print_summary(db: "Database") -> None:
        """Print the dispatch section of ``Database.print_summary``."""
        uarch_names = DispatchAnalyzer.uarchs_with_dispatch_data(db)
        if not uarch_names:
            return

        print("\n" + "═" * 70)
        print("  DISPATCH INSTRUMENTATION")
        print("    one row per dispatch size, pooled over all loaded TPGs")
        print("═" * 70)

        for name in uarch_names:
            samples = DispatchAnalyzer.collect_samples(db, name)
            stats = DispatchAnalyzer.size_stats(samples)
            deltas = dict(DispatchAnalyzer.marginal_costs(samples))
            n_tpgs = len({s.tpg_source_path for s in samples})
            print(f"\n  uarch: {name}   ({len(samples)} samples "
                  f"from {n_tpgs} TPG(s))")
            header = (f"    {'size':>4} {'nTPG':>5} {'events':>8} "
                      f"{'mean':>9} {'min':>8} {'max':>8} {'Δ/size':>8}")
            print(header)
            print("    " + "-" * (len(header) - 4))
            for st in stats:
                d = deltas.get(st.dispatch_size)
                d_str = f"{d:>+8.2f}" if d is not None else f"{'—':>8}"
                mark = "*" if st.is_sparse else " "
                print(f"    {st.dispatch_size:>4} {st.n_samples:>5} "
                      f"{st.n_events:>8} {st.mean_latency:>9.2f} "
                      f"{st.min_latency:>8.1f} {st.max_latency:>8.1f} "
                      f"{d_str} {mark}")
            bp = DispatchAnalyzer.suggest_breakpoint(samples)
            if bp is not None:
                print(f"    Δ/size jumps at size {bp} — consider "
                      f"`train-dispatch --breakpoint {bp}`")
        print("═" * 70)


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

def _fit_estimator(X, y, w, robust: bool, weighted: bool):
    """Fit one linear model (Huber or least squares) on a design matrix."""
    from sklearn.linear_model import HuberRegressor, LinearRegression
    if robust:
        # Huber: squared loss on small residuals, linear beyond epsilon, so a
        # contaminated aggregate cannot pull the line towards itself.
        # alpha=0 keeps the coefficients unregularised and comparable to LS.
        model = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=1000)
    else:
        model = LinearRegression()
    model.fit(X, y, sample_weight=w if weighted else None)
    return model


def _candidate_breakpoints(sizes: list[int], min_segment: int) -> list[int]:
    """Sizes with at least *min_segment* distinct sizes strictly below and at/above."""
    return [k for k in sizes
            if len([s for s in sizes if s < k]) >= min_segment
            and len([s for s in sizes if s >= k]) >= min_segment]


class DispatchRegressor:
    """
    Fits :class:`DispatchModel` — a (weighted) least-squares regression of
    dispatch latency on dispatch size.

    Plain least squares is used rather than the LassoCV of the team-level
    pipeline: there are only a handful of features here, so there is nothing
    to select and regularisation would only bias the coefficients.
    """

    @staticmethod
    def train(
        samples:      list[DispatchSample],
        degree:       int   = 1,
        breakpoint:   "int | str | None" = None,
        weighted:     bool  = True,
        robust:       bool  = False,
        test_size:    float = 0.20,
        random_state: int   = 42,
        min_segment:  int   = 2,
        cv_splits:    int   = 5,
        select_margin: float = 0.0,
        n_bootstrap:  int   = 0,
    ) -> DispatchModel:
        """
        Fit a dispatch model on *samples*.

        Args:
            samples:      One DispatchSample per (TPG, uarch, size).
            degree:       Polynomial degree in dispatch size (1 = affine).
            breakpoint:   Size at which the marginal cost is allowed to change,
                          adding a level shift and a slope change there.  None
                          fits a single regime.  ``"auto"`` selects it by
                          grouped cross-validation *inside the training fold*,
                          so the held-out fold influences neither the
                          coefficients nor the structure of the model.
            weighted:     Weight each sample by its event count, so aggregates
                          averaged over more dispatch events carry more
                          influence.
            robust:       Fit with a Huber loss instead of squared error, so a
                          handful of contaminated aggregates cannot dominate
                          the coefficients.  Samples it judges outlying are
                          reported rather than silently dropped.
            test_size:    Fraction held out for evaluation.
            random_state: Seed for the split (reproducibility).
            min_segment:  Distinct sizes required on each side of a candidate
                          breakpoint.  Two is the identifiability minimum; a
                          segment with exactly two sizes has no residual
                          degrees of freedom, so its linearity is untested.
            cv_splits:    Folds used by the automatic selection.
            select_margin: CV MAE improvement a breakpoint must achieve to be
                          preferred over a single regime.
            n_bootstrap:  Resamples for a bootstrap clustered on TPGs; 0 skips
                          it.  Gives confidence intervals for the reported
                          quantities and, under ``"auto"``, how often each
                          breakpoint is re-selected.

        Returns:
            A fitted :class:`DispatchModel`.

        Raises:
            ValueError:  If the basis is not identifiable from the sizes present.
            ImportError: If scikit-learn / numpy are not installed.
        """
        if len(samples) < 2:
            raise ValueError(
                f"Need at least 2 dispatch samples to train, got {len(samples)}."
            )
        distinct_sizes = sorted({s.dispatch_size for s in samples})
        if len(distinct_sizes) < 2:
            raise ValueError(
                f"All dispatch samples share the same DispatchSize "
                f"({distinct_sizes[0]}); the size coefficient is not "
                "identifiable.  Load more TPGs with varied dispatch sizes."
            )
        if degree < 1:
            raise ValueError("degree must be >= 1")
        if degree >= len(distinct_sizes):
            raise ValueError(
                f"degree {degree} needs more than {degree} distinct dispatch "
                f"sizes, only {len(distinct_sizes)} present ({distinct_sizes})."
            )
        auto_breakpoint = (breakpoint == "auto")
        if breakpoint is not None and not auto_breakpoint:
            below = [s for s in distinct_sizes if s < breakpoint]
            above = [s for s in distinct_sizes if s >= breakpoint]
            if len(below) < min_segment or len(above) < min_segment:
                raise ValueError(
                    f"breakpoint {breakpoint} needs at least {min_segment} "
                    f"distinct sizes on each side; got {below} below and "
                    f"{above} at/above.  Available sizes: {distinct_sizes}."
                )
        if auto_breakpoint and not _candidate_breakpoints(distinct_sizes,
                                                          min_segment):
            print(f"[DispatchRegressor] no candidate breakpoint has "
                  f"{min_segment} distinct sizes on each side of it "
                  f"({distinct_sizes}) — fitting a single regime.")
            breakpoint = None
            auto_breakpoint = False

        try:
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import (mean_absolute_error,
                                         mean_squared_error, r2_score)
        except ImportError as exc:
            raise ImportError(
                "scikit-learn and numpy are required for DispatchRegressor.train()"
            ) from exc

        # Under "auto" the breakpoint is unknown until the selection has run
        # on the training fold, so start from the single-regime design and
        # rebuild once it is known.
        if auto_breakpoint:
            breakpoint = None
        names = _feature_names(degree, breakpoint)
        X = np.array([_design_row(s.dispatch_size, degree, breakpoint)
                      for s in samples])
        y = np.array([s.latency for s in samples], dtype=float)
        w = np.array([max(s.nb_measurements, 1) for s in samples], dtype=float)
        sizes = np.array([s.dispatch_size for s in samples], dtype=float)

        # ── 80 / 20 split ────────────────────────────────────────────────
        # Below 5 samples a held-out fold is meaningless; fit on everything
        # and say so, mirroring the team-level Regressor.
        idx = np.arange(len(samples))
        if len(samples) < 5:
            print(f"[DispatchRegressor] Warning: only {len(samples)} samples — "
                  f"skipping train/test split, fitting on full dataset.")
            idx_train, idx_test = idx, idx
            n_test = 0
        else:
            idx_train, idx_test = train_test_split(
                idx, test_size=test_size, random_state=random_state,
            )
            n_test = len(idx_test)

        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        w_train = w[idx_train]
        sz_train, sz_test = sizes[idx_train], sizes[idx_test]

        # ── Breakpoint selection — training fold only ────────────────────
        # Run after the split, on the training samples alone, so the held-out
        # fold influences neither the coefficients nor the model's structure.
        selection: "BreakpointSelection | None" = None
        if auto_breakpoint:
            train_samples = [samples[i] for i in idx_train]
            selection = DispatchAnalyzer.select_breakpoint(
                train_samples, degree=degree, weighted=weighted, robust=robust,
                min_segment=min_segment, n_splits=cv_splits,
                margin=select_margin,
            )
            breakpoint = selection.chosen
            # The design matrix depends on the breakpoint, so rebuild it.
            names = _feature_names(degree, breakpoint)
            X = np.array([_design_row(s.dispatch_size, degree, breakpoint)
                          for s in samples])
            X_train, X_test = X[idx_train], X[idx_test]

        model = _fit_estimator(X_train, y_train, w_train, robust, weighted)

        def _metrics(y_true, y_pred) -> RegressionMetrics:
            return RegressionMetrics(
                mae  = float(mean_absolute_error(y_true, y_pred)),
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))),
                r2   = float(r2_score(y_true, y_pred)),
            )

        # ── Saturated baseline ───────────────────────────────────────────
        # Each size predicted by its own weighted mean over the *training*
        # fold: the best a model seeing only the size could ever do.  Sizes
        # absent from the training fold fall back to the global mean.
        global_mean = float(np.average(y_train, weights=w_train))
        size_means: dict[float, float] = {}
        for size in np.unique(sz_train):
            mask = sz_train == size
            size_means[float(size)] = float(
                np.average(y_train[mask], weights=w_train[mask])
            )

        def _saturated(sz) -> "np.ndarray":
            return np.array([size_means.get(float(s), global_mean) for s in sz])

        weights = {name: float(c) for name, c in zip(names, model.coef_)}

        # Per-size breakdown over all samples, with the model's prediction.
        stats = DispatchAnalyzer.size_stats(samples)
        for st in stats:
            st.predicted = float(model.predict(
                np.array([_design_row(st.dispatch_size, degree, breakpoint)])
            )[0])

        # Samples the fit cannot account for: residual greater than three
        # times the typical error.  Reported for both estimators — under
        # least squares these are the points distorting the coefficients,
        # under Huber they are the points it successfully resisted.
        resid_train = y_train - model.predict(X_train)
        threshold = 3.0 * float(np.mean(np.abs(resid_train)))
        flagged = [samples[i] for i, r in zip(idx_train, resid_train)
                   if abs(r) > threshold] if threshold > 0 else []

        sems = np.array([s.sem for s in samples], dtype=float)
        noise_floor = float(np.sqrt(np.mean(sems ** 2))) if len(sems) else 0.0

        # ── Segments resting on the bare identifiability minimum ─────────
        thin: list[str] = []
        if breakpoint is not None:
            below = [s for s in distinct_sizes if s < breakpoint]
            above = [s for s in distinct_sizes if s >= breakpoint]
            for label, seg in (("below", below), ("at/above", above)):
                if len(seg) == 2:
                    thin.append(
                        f"the segment {label} size {breakpoint} has only 2 "
                        f"distinct sizes {seg}; its line passes exactly "
                        f"through both means, so its linearity is untested")

        # ── Cluster bootstrap over TPGs ──────────────────────────────────
        # Samples from one TPG share a binary and are correlated, so the
        # resampling unit is the TPG, not the sample.  Resampling samples
        # independently would treat correlated observations as independent
        # evidence and give intervals that are far too narrow.
        intervals: list[BootstrapInterval] = []
        bp_counts: dict = {}
        if n_bootstrap > 0:
            rng = np.random.default_rng(random_state)
            train_samples = [samples[i] for i in idx_train]
            by_tpg: dict[str, list[DispatchSample]] = {}
            for smp in train_samples:
                by_tpg.setdefault(smp.tpg_source_path, []).append(smp)
            tpg_keys = list(by_tpg)

            draws: dict[str, list[float]] = {}
            for _ in range(n_bootstrap):
                picked = rng.choice(len(tpg_keys), size=len(tpg_keys),
                                    replace=True)
                boot = [smp for i in picked for smp in by_tpg[tpg_keys[i]]]
                boot_sizes = sorted({s.dispatch_size for s in boot})

                bp_b = breakpoint
                if auto_breakpoint:
                    if not _candidate_breakpoints(boot_sizes, min_segment):
                        bp_b = None
                    else:
                        sel_b = DispatchAnalyzer.select_breakpoint(
                            boot, degree=degree, weighted=weighted,
                            robust=robust, min_segment=min_segment,
                            n_splits=cv_splits, margin=select_margin,
                        )
                        bp_b = sel_b.chosen
                bp_counts[bp_b] = bp_counts.get(bp_b, 0) + 1

                Xb = np.array([_design_row(s.dispatch_size, degree, bp_b)
                               for s in boot])
                yb = np.array([s.latency for s in boot], dtype=float)
                wb = np.array([max(s.nb_measurements, 1) for s in boot],
                              dtype=float)
                if len(boot) <= Xb.shape[1]:
                    continue
                try:
                    mb = _fit_estimator(Xb, yb, wb, robust, weighted)
                except Exception:
                    continue
                nb = _feature_names(degree, bp_b)
                wmap = dict(zip(nb, (float(c) for c in mb.coef_)))
                draws.setdefault("bias", []).append(float(mb.intercept_))
                draws.setdefault("slope (low regime)", []).append(
                    wmap.get("dispatch_size", float("nan")))
                if bp_b is not None:
                    draws.setdefault("slope (high regime)", []).append(
                        wmap.get("dispatch_size", 0.0)
                        + wmap.get(f"extra_slope_ge{bp_b}", 0.0))
                    draws.setdefault("step at breakpoint", []).append(
                        wmap.get(f"step_ge{bp_b}", 0.0)
                        + wmap.get(f"extra_slope_ge{bp_b}", 0.0))

            point = {
                "bias": float(model.intercept_),
                "slope (low regime)": weights.get("dispatch_size", 0.0),
            }
            if breakpoint is not None:
                point["slope (high regime)"] = (
                    weights.get("dispatch_size", 0.0)
                    + weights.get(f"extra_slope_ge{breakpoint}", 0.0))
                point["step at breakpoint"] = (
                    weights.get(f"step_ge{breakpoint}", 0.0)
                    + weights.get(f"extra_slope_ge{breakpoint}", 0.0))
            for name, vals in draws.items():
                # Only report quantities the fitted model actually has: on a
                # single-regime fit some resamples still select a breakpoint,
                # but a "step" interval would be meaningless there.
                if name not in point:
                    continue
                arr = np.array(vals, dtype=float)
                arr = arr[~np.isnan(arr)]
                if arr.size < 2:
                    continue
                intervals.append(BootstrapInterval(
                    name=name,
                    point=point.get(name, float(np.median(arr))),
                    low=float(np.percentile(arr, 2.5)),
                    high=float(np.percentile(arr, 97.5)),
                ))

        return DispatchModel(
            weights       = weights,
            bias          = float(model.intercept_),
            degree        = degree,
            breakpoint    = breakpoint,
            weighted      = weighted,
            feature_names = names,
            split         = TrainTestSplit(
                n_total      = len(samples),
                n_train      = len(y_train),
                n_test       = n_test,
                random_state = random_state,
            ),
            train_metrics = _metrics(y_train, model.predict(X_train)),
            test_metrics  = (_metrics(y_test, model.predict(X_test))
                             if n_test > 0 else None),
            saturated_train_metrics = _metrics(y_train, _saturated(sz_train)),
            saturated_test_metrics  = (_metrics(y_test, _saturated(sz_test))
                                       if n_test > 0 else None),
            size_stats    = stats,
            noise_floor   = noise_floor,
            estimator     = "huber" if robust else "least squares",
            flagged_outliers = flagged,
            selection     = selection,
            bootstrap     = intervals,
            bootstrap_breakpoints = bp_counts,
            thin_segments = thin,
        )

    @staticmethod
    def predict(model: DispatchModel, dispatch_size: float) -> float:
        """Predict the cost of one dispatch of the given size."""
        return model.predict(dispatch_size)