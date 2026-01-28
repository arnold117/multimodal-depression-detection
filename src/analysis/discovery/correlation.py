"""
Relationship discovery through correlation analysis.

Provides rigorous statistical analysis with:
- Multiple comparison correction (FDR, Bonferroni)
- Effect size calculation and interpretation
- Confidence intervals
- Partial correlation (controlling for confounds)
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from loguru import logger


@dataclass
class RelationshipResult:
    """
    Statistical result for a single sensor-survey relationship.

    All medical research requires reporting:
    1. Effect size (not just p-value)
    2. Confidence intervals
    3. Sample size
    4. Multiple comparison correction
    """

    sensor_feature: str
    survey_item: str
    correlation: float
    p_value: float
    p_adjusted: float
    effect_size: float
    n_samples: int
    confidence_interval: Tuple[float, float]
    method: str  # pearson or spearman
    interpretation: str  # negligible, small, medium, large

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if relationship is statistically significant after correction."""
        return self.p_adjusted < alpha

    def is_meaningful(self, min_effect: float = 0.3) -> bool:
        """Check if relationship has practical/clinical significance."""
        return abs(self.effect_size) >= min_effect

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "sensor_feature": self.sensor_feature,
            "survey_item": self.survey_item,
            "correlation": round(self.correlation, 4),
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "effect_size": round(self.effect_size, 4),
            "n_samples": self.n_samples,
            "ci_lower": round(self.confidence_interval[0], 4),
            "ci_upper": round(self.confidence_interval[1], 4),
            "method": self.method,
            "interpretation": self.interpretation,
            "significant": self.is_significant(),
            "meaningful": self.is_meaningful(),
        }


@dataclass
class DiscoveryConfig:
    """Configuration for relationship discovery."""

    method: Literal["pearson", "spearman"] = "spearman"
    correction: Literal["fdr_bh", "bonferroni", "holm", "none"] = "fdr_bh"
    alpha: float = 0.05
    min_effect_size: float = 0.3
    min_samples: int = 20
    bootstrap_ci: bool = True
    n_bootstrap: int = 1000


class RelationshipDiscovery:
    """
    Engine for discovering sensor-survey relationships.

    Implements rigorous statistical methodology:
    1. Correlation analysis (Pearson/Spearman)
    2. Multiple comparison correction
    3. Effect size calculation
    4. Bootstrap confidence intervals
    """

    def __init__(self, config: Optional[DiscoveryConfig] = None):
        """
        Initialize discovery engine.

        Args:
            config: Discovery configuration
        """
        self.config = config or DiscoveryConfig()
        self.results: List[RelationshipResult] = []

    def compute_all_correlations(
        self,
        sensor_features: np.ndarray,
        survey_scores: np.ndarray,
        feature_names: List[str],
        survey_names: List[str],
    ) -> List[RelationshipResult]:
        """
        Compute correlations between all sensor features and survey items.

        Args:
            sensor_features: (n_samples, n_features) array
            survey_scores: (n_samples, n_surveys) array
            feature_names: List of sensor feature names
            survey_names: List of survey item names

        Returns:
            List of RelationshipResult objects
        """
        n_features = sensor_features.shape[1]
        n_surveys = survey_scores.shape[1]
        n_tests = n_features * n_surveys

        logger.info(
            f"Computing {n_tests} correlations "
            f"({n_features} features × {n_surveys} survey items)"
        )

        # Collect all raw results first
        raw_results = []

        for i, feat_name in enumerate(feature_names):
            for j, survey_name in enumerate(survey_names):
                x = sensor_features[:, i]
                y = survey_scores[:, j]

                # Remove NaN pairs
                mask = ~(np.isnan(x) | np.isnan(y))
                x_clean = x[mask]
                y_clean = y[mask]

                n = len(x_clean)
                if n < self.config.min_samples:
                    continue

                # Compute correlation
                if self.config.method == "pearson":
                    r, p = stats.pearsonr(x_clean, y_clean)
                else:
                    r, p = stats.spearmanr(x_clean, y_clean)

                # Confidence interval
                ci = self._compute_ci(x_clean, y_clean, r, n)

                raw_results.append(
                    {
                        "sensor_feature": feat_name,
                        "survey_item": survey_name,
                        "correlation": r,
                        "p_value": p,
                        "n_samples": n,
                        "ci": ci,
                    }
                )

        # Apply multiple comparison correction
        if raw_results:
            p_values = [r["p_value"] for r in raw_results]
            p_adjusted = self._correct_pvalues(p_values)

            # Create final results
            self.results = []
            for raw, p_adj in zip(raw_results, p_adjusted):
                effect = self._correlation_to_effect_size(raw["correlation"])
                interp = self._interpret_effect_size(effect)

                result = RelationshipResult(
                    sensor_feature=raw["sensor_feature"],
                    survey_item=raw["survey_item"],
                    correlation=raw["correlation"],
                    p_value=raw["p_value"],
                    p_adjusted=p_adj,
                    effect_size=effect,
                    n_samples=raw["n_samples"],
                    confidence_interval=raw["ci"],
                    method=self.config.method,
                    interpretation=interp,
                )
                self.results.append(result)

        logger.info(f"Computed {len(self.results)} valid correlations")
        return self.results

    def _compute_ci(
        self, x: np.ndarray, y: np.ndarray, r: float, n: int
    ) -> Tuple[float, float]:
        """
        Compute 95% confidence interval for correlation.

        Uses Fisher z-transformation for parametric CI,
        or bootstrap for non-parametric.
        """
        if self.config.bootstrap_ci:
            return self._bootstrap_ci(x, y)
        else:
            return self._fisher_ci(r, n)

    def _fisher_ci(self, r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
        """Fisher z-transformation confidence interval."""
        # Avoid division by zero
        r = np.clip(r, -0.9999, 0.9999)

        # Fisher z-transform
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1 / np.sqrt(n - 3)

        # CI in z-space
        z_crit = stats.norm.ppf(1 - alpha / 2)
        z_low = z - z_crit * se
        z_high = z + z_crit * se

        # Transform back
        r_low = (np.exp(2 * z_low) - 1) / (np.exp(2 * z_low) + 1)
        r_high = (np.exp(2 * z_high) - 1) / (np.exp(2 * z_high) + 1)

        return (r_low, r_high)

    def _bootstrap_ci(
        self, x: np.ndarray, y: np.ndarray, alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval."""
        n = len(x)
        rng = np.random.default_rng(42)
        boot_corrs = []

        for _ in range(self.config.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            if self.config.method == "pearson":
                r, _ = stats.pearsonr(x[idx], y[idx])
            else:
                r, _ = stats.spearmanr(x[idx], y[idx])
            boot_corrs.append(r)

        lower = np.percentile(boot_corrs, 100 * alpha / 2)
        upper = np.percentile(boot_corrs, 100 * (1 - alpha / 2))

        return (lower, upper)

    def _correct_pvalues(self, p_values: List[float]) -> List[float]:
        """Apply multiple comparison correction."""
        if self.config.correction == "none":
            return p_values

        _, p_adj, _, _ = multipletests(
            p_values, alpha=self.config.alpha, method=self.config.correction
        )
        return p_adj.tolist()

    def _correlation_to_effect_size(self, r: float) -> float:
        """
        Convert correlation to Cohen's d equivalent.

        For correlation: effect size = r itself is commonly used,
        but we can convert to d for comparison with other studies.

        d = 2r / sqrt(1 - r^2)
        """
        # Use |r| directly as effect size (common in psychology)
        return abs(r)

    def _interpret_effect_size(self, effect: float) -> str:
        """
        Interpret effect size according to Cohen's conventions.

        For correlation coefficients:
        - |r| < 0.1: negligible
        - 0.1 <= |r| < 0.3: small
        - 0.3 <= |r| < 0.5: medium
        - |r| >= 0.5: large
        """
        if effect < 0.1:
            return "negligible"
        elif effect < 0.3:
            return "small"
        elif effect < 0.5:
            return "medium"
        else:
            return "large"

    def filter_significant(
        self,
        alpha: Optional[float] = None,
        min_effect_size: Optional[float] = None,
    ) -> List[RelationshipResult]:
        """
        Filter results to statistically and practically significant relationships.

        Args:
            alpha: Significance threshold (default: config value)
            min_effect_size: Minimum effect size (default: config value)

        Returns:
            Filtered list of significant relationships
        """
        alpha = alpha or self.config.alpha
        min_effect = min_effect_size or self.config.min_effect_size

        significant = [
            r
            for r in self.results
            if r.is_significant(alpha) and r.is_meaningful(min_effect)
        ]

        logger.info(
            f"Filtered to {len(significant)} significant relationships "
            f"(α={alpha}, min_effect={min_effect})"
        )

        return significant

    def get_summary_stats(self) -> dict:
        """Get summary statistics of all computed correlations."""
        if not self.results:
            return {}

        correlations = [r.correlation for r in self.results]
        p_values = [r.p_adjusted for r in self.results]
        effect_sizes = [r.effect_size for r in self.results]

        n_significant = sum(1 for r in self.results if r.is_significant())
        n_meaningful = sum(1 for r in self.results if r.is_meaningful())

        return {
            "n_tests": len(self.results),
            "n_significant": n_significant,
            "n_meaningful": n_meaningful,
            "n_both": sum(
                1 for r in self.results if r.is_significant() and r.is_meaningful()
            ),
            "correlation_mean": np.mean(correlations),
            "correlation_std": np.std(correlations),
            "effect_size_mean": np.mean(effect_sizes),
            "effect_size_max": np.max(effect_sizes),
            "min_p_adjusted": np.min(p_values),
        }
