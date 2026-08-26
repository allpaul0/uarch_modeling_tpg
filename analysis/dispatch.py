"""
analysis/dispatch.py — modelling the TPG dispatch from its size.

The dispatch is the control step that runs *after* a Team has executed: the
programs' results are compared, the largest one wins, and control moves to the
winner's destination (another Team, or an Action).  It is instrumented as a
whole in ``inferenceTPG`` and reported per ``DispatchSize`` in
``latencies.json``, so — unlike Teams — there is no instruction stream to
extract features from.

The model is therefore a small, explicit one::

    cycles(dispatch) ≈ bias + w1·size [+ w2·size² …]

where ``bias`` is the fixed dispatch overhead (probe, bookkeeping, the final
jump) and ``w1`` is the marginal cost of comparing one more program.  Samples
are pooled across every TPG and every loaded folder in the Database, exactly
like the team-level pipeline: one sample = one (TPG, uarch, dispatch size)
aggregate.

Two details matter for getting sensible numbers out of it:

*   Each sample is already an average over ``Count`` dispatch events, so its
    precision varies a lot (Count ranges from 10 to 100 in practice).  Fitting
    is weighted by ``Count`` by default so that well-measured points count for
    more — this is ordinary weighted least squares, not a heuristic.
*   Whether the relation is actually linear in size is an empirical question.
    :meth:`DispatchModel.print_report` prints a per-size residual table so
    curvature is visible; ``degree=2`` fits a quadratic if it is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from analysis.regression import RegressionMetrics, TrainTestSplit

if TYPE_CHECKING:
    from classes.database import Database
    from classes.tpg import TPG


# Feature names used by the polynomial basis, in order.
def _feature_names(degree: int) -> list[str]:
    names = ["dispatch_size"]
    for d in range(2, degree + 1):
        names.append(f"dispatch_size^{d}")
    return names


def _design_row(size: float, degree: int) -> list[float]:
    return [float(size) ** d for d in range(1, degree + 1)]


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
    dispatch_size:   Number of programs compared — the only feature.
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


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

@dataclass
class DispatchModel:
    """
    A fitted dispatch model: ``cycles ≈ bias + Σ w_d · size^d``.

    Attributes
    ----------
    weights:       Coefficient per basis feature ("dispatch_size", …).
    bias:          Intercept — the size-independent dispatch overhead.
    degree:        Polynomial degree used (1 = affine in size).
    weighted:      Whether samples were weighted by their event count.
    feature_names: Basis feature names, in coefficient order.
    split:         Train/test split information.
    train_metrics: MAE / RMSE / R² on the training fold.
    test_metrics:  MAE / RMSE / R² on the held-out fold.
    size_stats:    Per-size aggregates with the model's prediction, for
                   inspecting linearity.
    noise_floor:   RMSE that pure measurement noise alone would produce
                   (root-mean-square of the per-sample standard errors).  A
                   test RMSE near this value means the model is as good as the
                   data allows; far above it means real structure is missing.
    """
    weights:       dict[str, float]
    bias:          float
    degree:        int = 1
    weighted:      bool = True
    feature_names: list[str] = field(default_factory=list)
    split:         TrainTestSplit | None = None
    train_metrics: RegressionMetrics | None = None
    test_metrics:  RegressionMetrics | None = None
    size_stats:    list[DispatchSizeStat] = field(default_factory=list)
    noise_floor:   float = 0.0

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(self, dispatch_size: float) -> float:
        """Predicted cycle cost of a dispatch comparing *dispatch_size* programs."""
        total = self.bias
        for name, value in zip(_feature_names(self.degree),
                               _design_row(dispatch_size, self.degree)):
            total += self.weights.get(name, 0.0) * value
        return total

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #

    def formula(self) -> str:
        """Human-readable formula, e.g. "cycles ≈ 1.83 + 15.62·size"."""
        parts = [f"{self.bias:.2f}"]
        for name in _feature_names(self.degree):
            w = self.weights.get(name, 0.0)
            exponent = name.split("^")[1] if "^" in name else "1"
            term = "size" if exponent == "1" else f"size^{exponent}"
            parts.append(f"{'+' if w >= 0 else '-'} {abs(w):.2f}·{term}")
        return "cycles ≈ " + " ".join(parts)

    def __repr__(self) -> str:
        return (f"DispatchModel({self.formula()}, degree={self.degree}, "
                f"weighted={self.weighted})")

    def print_report(self) -> None:
        """Print a human-readable dispatch-regression report."""
        bar = "─" * 72
        print(f"\n{bar}")
        print("  DISPATCH REGRESSION REPORT")
        print(bar)
        if self.split:
            s = self.split
            print(f"  Dataset split (seed={s.random_state}): "
                  f"{s.n_train} train  /  {s.n_test} test  (total {s.n_total})")
        print(f"  Basis: degree {self.degree}"
              f"   weighting: {'by event count' if self.weighted else 'none'}")
        print(f"  Model: {self.formula()}")
        print(f"  Fixed overhead (bias): {self.bias:.3f} cycles")
        for name in _feature_names(self.degree):
            print(f"  Coefficient {name:<18} {self.weights.get(name, 0.0):+.4f}")

        if self.train_metrics:
            print(f"\n  Train metrics:  {self.train_metrics}")
        if self.test_metrics:
            print(f"  Test  metrics:  {self.test_metrics}")
        if self.noise_floor:
            print(f"  Noise floor  :  RMSE={self.noise_floor:.3f}  "
                  f"(measurement error alone; a test RMSE near this means the "
                  f"model is as accurate as the data permits)")

        if self.size_stats:
            print(f"\n  Per-size breakdown:")
            header = (f"    {'size':>4} {'n':>4} {'events':>7} "
                      f"{'measured':>10} {'predicted':>10} {'residual':>10} "
                      f"{'min..max':>17}")
            print(header)
            print("    " + "-" * (len(header) - 4))
            for st in self.size_stats:
                span = f"{st.min_latency:.0f}..{st.max_latency:.0f}"
                print(f"    {st.dispatch_size:>4} {st.n_samples:>4} "
                      f"{st.n_events:>7} {st.mean_latency:>10.2f} "
                      f"{st.predicted:>10.2f} {st.residual:>+10.2f} "
                      f"{span:>17}")
        print(bar)


# --------------------------------------------------------------------------- #
# Collection + fitting
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
                for size in sorted(tpg.get_dispatches(name)):
                    dm = tpg.get_dispatches(name)[size]
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
    def uarchs_with_dispatch_data(db: "Database") -> list[str]:
        """Uarch names for which at least one TPG has dispatch measurements."""
        names: set[str] = set()
        for tpg in db.tpgs.values():
            names.update(n for n, d in tpg.dispatch_latencies.items() if d)
        return sorted(names)

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
            n_tpgs = len({s.tpg_source_path for s in samples})
            print(f"\n  uarch: {name}   ({len(samples)} samples "
                  f"from {n_tpgs} TPG(s))")
            header = (f"    {'size':>4} {'nTPG':>5} {'events':>8} "
                      f"{'mean':>9} {'min':>8} {'max':>8}")
            print(header)
            print("    " + "-" * (len(header) - 4))
            for st in stats:
                print(f"    {st.dispatch_size:>4} {st.n_samples:>5} "
                      f"{st.n_events:>8} {st.mean_latency:>9.2f} "
                      f"{st.min_latency:>8.1f} {st.max_latency:>8.1f}")
        print("═" * 70)

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


class DispatchRegressor:
    """
    Fits :class:`DispatchModel` — a (weighted) least-squares regression of
    dispatch latency on dispatch size.

    Plain least squares is used rather than the LassoCV of the team-level
    pipeline: there are only one or two features here, so there is nothing to
    select and regularisation would only bias the coefficients.
    """

    @staticmethod
    def train(
        samples:      list[DispatchSample],
        degree:       int   = 1,
        weighted:     bool  = True,
        test_size:    float = 0.20,
        random_state: int   = 42,
    ) -> DispatchModel:
        """
        Fit a dispatch model on *samples*.

        Args:
            samples:      One DispatchSample per (TPG, uarch, size).
            degree:       Polynomial degree in dispatch size (1 = affine).
            weighted:     Weight each sample by its event count, so that
                          aggregates averaged over more dispatch events carry
                          more influence.
            test_size:    Fraction held out for evaluation.
            random_state: Seed for the split (reproducibility).

        Returns:
            A fitted :class:`DispatchModel`.

        Raises:
            ValueError:  If there are too few samples, or every sample shares
                         the same dispatch size (nothing to regress on).
            ImportError: If scikit-learn / numpy are not installed.
        """
        if len(samples) < 2:
            raise ValueError(
                f"Need at least 2 dispatch samples to train, got {len(samples)}."
            )
        distinct_sizes = {s.dispatch_size for s in samples}
        if len(distinct_sizes) < 2:
            raise ValueError(
                "All dispatch samples share the same DispatchSize "
                f"({distinct_sizes.pop()}); the size coefficient is not "
                "identifiable.  Load more TPGs with varied dispatch sizes."
            )
        if degree < 1:
            raise ValueError("degree must be >= 1")
        if degree >= len(distinct_sizes):
            raise ValueError(
                f"degree {degree} needs more than {degree} distinct dispatch "
                f"sizes, only {len(distinct_sizes)} present "
                f"({sorted(distinct_sizes)})."
            )

        try:
            import numpy as np
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import (mean_absolute_error,
                                         mean_squared_error, r2_score)
        except ImportError as exc:
            raise ImportError(
                "scikit-learn and numpy are required for DispatchRegressor.train()"
            ) from exc

        names = _feature_names(degree)
        X = np.array([_design_row(s.dispatch_size, degree) for s in samples])
        y = np.array([s.latency for s in samples], dtype=float)
        w = np.array([max(s.nb_measurements, 1) for s in samples], dtype=float)

        # ── 80 / 20 split ────────────────────────────────────────────────
        # Below 5 samples a held-out fold is meaningless; fit on everything
        # and say so, mirroring the team-level Regressor.
        if len(samples) < 5:
            print(f"[DispatchRegressor] Warning: only {len(samples)} samples — "
                  f"skipping train/test split, fitting on full dataset.")
            X_train, X_test, y_train, y_test = X, X, y, y
            w_train = w
            n_test = 0
        else:
            (X_train, X_test, y_train, y_test,
             w_train, _w_test) = train_test_split(
                X, y, w, test_size=test_size, random_state=random_state,
            )
            n_test = len(y_test)

        model = LinearRegression()
        model.fit(X_train, y_train,
                  sample_weight=w_train if weighted else None)

        def _metrics(X_fold, y_fold) -> RegressionMetrics:
            y_pred = model.predict(X_fold)
            return RegressionMetrics(
                mae  = float(mean_absolute_error(y_fold, y_pred)),
                rmse = float(np.sqrt(mean_squared_error(y_fold, y_pred))),
                r2   = float(r2_score(y_fold, y_pred)),
            )

        weights = {name: float(c) for name, c in zip(names, model.coef_)}

        # Per-size breakdown, with the model's prediction alongside.
        stats = DispatchAnalyzer.size_stats(samples)
        for st in stats:
            st.predicted = float(
                model.predict(np.array([_design_row(st.dispatch_size, degree)]))[0]
            )

        # How much RMSE the measurement noise alone accounts for.
        sems = np.array([s.sem for s in samples], dtype=float)
        noise_floor = float(np.sqrt(np.mean(sems ** 2))) if len(sems) else 0.0

        return DispatchModel(
            weights       = weights,
            bias          = float(model.intercept_),
            degree        = degree,
            weighted      = weighted,
            feature_names = names,
            split         = TrainTestSplit(
                n_total      = len(samples),
                n_train      = len(y_train),
                n_test       = n_test,
                random_state = random_state,
            ),
            train_metrics = _metrics(X_train, y_train),
            test_metrics  = _metrics(X_test, y_test) if n_test > 0 else None,
            size_stats    = stats,
            noise_floor   = noise_floor,
        )

    @staticmethod
    def predict(model: DispatchModel, dispatch_size: float) -> float:
        """Predict the cost of one dispatch of the given size."""
        return model.predict(dispatch_size)