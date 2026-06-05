from __future__ import annotations
from dataclasses import dataclass, field

from classes.features import FeatureVector


@dataclass
class RegressionModel:
    """
    A trained linear regression model: latency ≈ bias + Σ(weight_i · feature_i).
    """
    weights: dict[str, float]    # feature_name → coefficient
    bias: float
    feature_names: list[str] = field(default_factory=list)  # ordered key list

    def predict(self, fv: FeatureVector) -> float:
        """
        Predict latency for a single FeatureVector.

        Args:
            fv: Feature vector (keys need not be a superset of trained features;
                missing keys contribute 0).

        Returns:
            Estimated latency in cycles.
        """
        return self.bias + sum(
            self.weights.get(name, 0.0) * fv.get(name, 0.0)
            for name in self.feature_names
        )

    def __repr__(self) -> str:
        top = sorted(self.weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        top_str = ", ".join(f"{k}={v:.4f}" for k, v in top)
        return f"RegressionModel(bias={self.bias:.4f}, top_weights=[{top_str}])"


class Regressor:
    """
    Trains a linear regression model that maps FeatureVectors to latencies.

    Relies on scikit-learn under the hood; raises ImportError if not installed.
    """

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    @staticmethod
    def train(
        features: list[FeatureVector],
        latencies: list[float],
    ) -> RegressionModel:
        """
        Fit a linear regression model on (features, latencies) pairs.

        Args:
            features:  One FeatureVector per training sample.
            latencies: Corresponding measured latency values.

        Returns:
            A fitted RegressionModel.

        Raises:
            ValueError: If the two lists have different lengths or are empty.
            ImportError: If scikit-learn is not installed.
        """
        if len(features) != len(latencies):
            raise ValueError(
                f"features and latencies must have the same length "
                f"({len(features)} vs {len(latencies)})"
            )
        if not features:
            raise ValueError("Cannot train on an empty dataset.")

        try:
            import numpy as np
            from sklearn.linear_model import LinearRegression
        except ImportError as exc:
            raise ImportError(
                "scikit-learn and numpy are required for Regressor.train()"
            ) from exc

        # Build a consistent feature order from the union of all keys
        all_keys: list[str] = []
        seen: set[str] = set()
        for fv in features:
            for k in fv.feature_names():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        X = np.array([fv.as_list(all_keys) for fv in features])
        y = np.array(latencies)

        lr = LinearRegression()
        lr.fit(X, y)

        weights = {name: float(coef) for name, coef in zip(all_keys, lr.coef_)}
        return RegressionModel(
            weights=weights,
            bias=float(lr.intercept_),
            feature_names=all_keys,
        )

    # ------------------------------------------------------------------ #
    # Prediction (convenience — delegates to model)
    # ------------------------------------------------------------------ #

    @staticmethod
    def predict(model: RegressionModel, fv: FeatureVector) -> float:
        """
        Predict latency using a previously trained model.

        Args:
            model: A RegressionModel returned by train().
            fv:    Feature vector for a single team.

        Returns:
            Predicted latency.
        """
        return model.predict(fv)