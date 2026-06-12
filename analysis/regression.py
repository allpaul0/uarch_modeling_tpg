"""
analysis/regression.py — Lasso regression with 80/20 train/test split.

LassoCV is used instead of plain LinearRegression for two reasons:
  1. Lasso (L1 regularisation) drives irrelevant feature weights to exactly
     zero, giving a sparse, interpretable model — important when the feature
     space contains many bigram-transition features that may be noise.
  2. LassoCV selects the regularisation strength α automatically via
     cross-validation on the training fold, so no manual tuning is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


from classes.features import FeatureVector


@dataclass
class TrainTestSplit:
    """Indices and sizes of the 80/20 split used to train a model."""
    n_total:    int
    n_train:    int
    n_test:     int
    random_state: int


@dataclass
class RegressionMetrics:
    """Evaluation metrics computed on one fold (train or test)."""
    mae:  float    # Mean Absolute Error  (cycles)
    rmse: float    # Root Mean Squared Error  (cycles)
    r2:   float    # Coefficient of determination  (1 = perfect)

    def __repr__(self) -> str:
        return f"MAE={self.mae:.3f}  RMSE={self.rmse:.3f}  R²={self.r2:.4f}"


@dataclass
class RegressionModel:
    """
    A trained Lasso regression model: latency ≈ bias + Σ(weight_i · feature_i).

    Attributes
    ----------
    weights:        Non-zero feature coefficients after Lasso regularisation.
                    Features driven to zero by Lasso are absent from this dict.
    bias:           Intercept term.
    feature_names:  Ordered list of ALL features seen during training
                    (including zero-weight ones), needed to rebuild the input
                    vector correctly for future predictions.
    alpha:          Regularisation strength chosen by LassoCV.
    split:          Information about the 80/20 split used during training.
    train_metrics:  MAE / RMSE / R² on the 80 % training fold.
    test_metrics:   MAE / RMSE / R² on the held-out 20 % test fold.
    """
    weights:       dict[str, float]
    bias:          float
    feature_names: list[str]       = field(default_factory=list)
    alpha:         float           = 0.0
    split:         TrainTestSplit  | None = None
    train_metrics: RegressionMetrics | None = None
    test_metrics:  RegressionMetrics | None = None

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(self, fv: FeatureVector) -> float:
        """
        Predict latency for a single FeatureVector.

        Missing features contribute 0, so vectors from unseen teams work fine.
        """
        return self.bias + sum(
            self.weights.get(name, 0.0) * fv.get(name, 0.0)
            for name in self.feature_names
        )

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        n_nonzero = sum(1 for v in self.weights.values() if v != 0.0)
        top = sorted(
            ((k, v) for k, v in self.weights.items() if v != 0.0),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:5]
        top_str = ", ".join(f"{k}={v:.4f}" for k, v in top)
        return (
            f"RegressionModel("
            f"alpha={self.alpha:.6f}, "
            f"bias={self.bias:.4f}, "
            f"non_zero_features={n_nonzero}/{len(self.feature_names)}, "
            f"top_weights=[{top_str}])"
        )

    def print_report(self) -> None:
        """Print a human-readable training report."""
        bar = "─" * 60
        print(f"\n{bar}")
        print("  LASSO REGRESSION REPORT")
        print(bar)
        if self.split:
            s = self.split
            print(f"  Dataset split (seed={s.random_state}): "
                  f"{s.n_train} train  /  {s.n_test} test  "
                  f"(total {s.n_total})")
        print(f"  Alpha (L1 strength, via CV): {self.alpha:.6f}")
        n_nz = sum(1 for v in self.weights.values() if v != 0.0)
        print(f"  Features: {n_nz} non-zero  /  {len(self.feature_names)} total")
        print(f"  Bias (intercept):  {self.bias:.4f}")

        if self.train_metrics:
            print(f"\n  Train metrics:  {self.train_metrics}")
        if self.test_metrics:
            print(f"  Test  metrics:  {self.test_metrics}")

        if n_nz:
            top = sorted(
                ((k, v) for k, v in self.weights.items() if v != 0.0),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )
            print(f"\n  Non-zero weights ({n_nz}):")
            for k, v in top:
                bar_len = int(abs(v) / max(abs(w) for _, w in top) * 30)
                direction = "+" if v > 0 else "-"
                print(f"    {k:<40} {v:+.4f}  {direction*bar_len}")
        print(bar)


class Regressor:
    """
    Trains a Lasso regression model with an 80/20 train/test split.

    Uses :class:`sklearn.linear_model.LassoCV` which selects the
    regularisation strength α by cross-validation on the training fold.
    """

    @staticmethod
    def train(
        features:     list[FeatureVector],
        latencies:    list[float],
        test_size:    float = 0.20,
        random_state: int   = 42,
        cv:           int   = 5,
    ) -> RegressionModel:
        """
        Fit a Lasso model on *features* / *latencies* pairs.

        The dataset is split 80/20 (stratification is not applied because
        latency is continuous).  LassoCV selects α via *cv*-fold cross-
        validation on the training fold only — the test fold is never seen
        during fitting.

        Args:
            features:     One FeatureVector per sample.
            latencies:    Corresponding measured latency values (cycles).
            test_size:    Fraction of samples held out for evaluation
                          (default 0.20 = 20 %).
            random_state: Seed for the train/test split (reproducibility).
            cv:           Number of cross-validation folds for alpha search.

        Returns:
            A fitted :class:`RegressionModel` with train and test metrics.

        Raises:
            ValueError:   If inputs are inconsistent or too small.
            ImportError:  If scikit-learn / numpy are not installed.
        """
        if len(features) != len(latencies):
            raise ValueError(
                f"features and latencies must have the same length "
                f"({len(features)} vs {len(latencies)})"
            )
        if len(features) < 2:
            raise ValueError("Need at least 2 samples to train.")

        try:
            import numpy as np
            from sklearn.linear_model import LassoCV
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        except ImportError as exc:
            raise ImportError(
                "scikit-learn and numpy are required for Regressor.train()"
            ) from exc

        # ── Build a consistent feature matrix ────────────────────────────
        all_keys: list[str] = []
        seen: set[str] = set()
        for fv in features:
            for k in fv.feature_names():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        X = np.array([fv.as_list(all_keys) for fv in features])
        y = np.array(latencies, dtype=float)

        # ── 80 / 20 split ────────────────────────────────────────────────
        # With very small datasets (< 5 samples) skip the split so training
        # can still proceed, but warn the caller.
        if len(features) < 5:
            print(
                f"[Regressor] Warning: only {len(features)} samples — "
                "skipping train/test split, fitting on full dataset."
            )
            X_train, X_test, y_train, y_test = X, X, y, y
            actual_test_size = 0
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
            )
            actual_test_size = len(y_test)

        n_train, n_test = len(y_train), actual_test_size

        # ── Fit LassoCV on training fold only ────────────────────────────
        lasso = LassoCV(cv=min(cv, n_train), max_iter=10_000, random_state=random_state)
        lasso.fit(X_train, y_train)

        weights = {
            name: float(coef)
            for name, coef in zip(all_keys, lasso.coef_)
        }

        # ── Metrics ──────────────────────────────────────────────────────
        def _metrics(X_fold: "np.ndarray", y_fold: "np.ndarray") -> RegressionMetrics:
            y_pred = lasso.predict(X_fold)
            return RegressionMetrics(
                mae  = float(mean_absolute_error(y_fold, y_pred)),
                rmse = float(np.sqrt(mean_squared_error(y_fold, y_pred))),
                r2   = float(r2_score(y_fold, y_pred)),
            )

        train_metrics = _metrics(X_train, y_train)
        test_metrics  = _metrics(X_test,  y_test) if n_test > 0 else None

        return RegressionModel(
            weights       = weights,
            bias          = float(lasso.intercept_),
            feature_names = all_keys,
            alpha         = float(lasso.alpha_),
            split         = TrainTestSplit(
                n_total      = len(features),
                n_train      = n_train,
                n_test       = n_test,
                random_state = random_state,
            ),
            train_metrics = train_metrics,
            test_metrics  = test_metrics,
        )

    @staticmethod
    def predict(model: RegressionModel, fv: FeatureVector) -> float:
        """Predict latency for a single FeatureVector using a trained model."""
        return model.predict(fv)