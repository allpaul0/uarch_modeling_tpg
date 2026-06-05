from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class FeatureVector:
    """
    A fixed mapping of feature names → float values extracted from a Team's
    instructions, e.g. {"ADD_count": 3.0, "LOAD_count": 5.0, "RAW_hazards": 2.0}.
    """
    id_team: int
    values: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def get(self, feature: str, default: float = 0.0) -> float:
        return self.values.get(feature, default)

    def feature_names(self) -> list[str]:
        return list(self.values.keys())

    def as_list(self, feature_names: list[str]) -> list[float]:
        """
        Return values in the given feature order (filling 0.0 for missing keys).
        Useful to build a consistent input vector for the regressor.
        """
        return [self.values.get(name, 0.0) for name in feature_names]

    # ------------------------------------------------------------------ #
    # Optional pandas integration
    # ------------------------------------------------------------------ #

    def to_series(self):  # type: ignore[return]
        """Convert to a pandas Series (lazy import)."""
        try:
            import pandas as pd
            return pd.Series(self.values, name=self.id_team)
        except ImportError as exc:
            raise ImportError("pandas is required for to_series()") from exc

    def __repr__(self) -> str:
        return f"FeatureVector(id_team={self.id_team}, values={self.values})"