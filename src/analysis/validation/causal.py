"""
Causal relationship validation.

Provides methods to validate whether discovered correlations
have potential causal interpretations:
1. PC Algorithm - Constraint-based causal discovery
2. Granger Causality - Time-series causal inference
3. Partial Correlation - Control for confounders

IMPORTANT: These methods suggest causal hypotheses, not proof.
True causal claims require experimental intervention.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from loguru import logger

# Optional: causal-learn for PC algorithm
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    logger.warning("causal-learn not installed. PC algorithm unavailable.")


@dataclass
class CausalResult:
    """Result of causal validation for a relationship."""

    source: str  # Potential cause
    target: str  # Potential effect
    method: str  # 'pc', 'granger', or 'partial_corr'

    # PC algorithm results
    edge_exists: Optional[bool] = None
    edge_direction: Optional[str] = None  # 'forward', 'backward', 'bidirectional', 'none'

    # Granger causality results
    granger_f_stat: Optional[float] = None
    granger_p_value: Optional[float] = None
    optimal_lag: Optional[int] = None

    # Partial correlation results
    partial_corr: Optional[float] = None
    partial_p_value: Optional[float] = None
    controlled_vars: Optional[List[str]] = None

    # Interpretation
    causal_evidence: str = "none"  # 'none', 'weak', 'moderate', 'strong'
    interpretation: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "method": self.method,
            "edge_exists": self.edge_exists,
            "edge_direction": self.edge_direction,
            "granger_f_stat": self.granger_f_stat,
            "granger_p_value": self.granger_p_value,
            "optimal_lag": self.optimal_lag,
            "partial_corr": self.partial_corr,
            "partial_p_value": self.partial_p_value,
            "controlled_vars": self.controlled_vars,
            "causal_evidence": self.causal_evidence,
            "interpretation": self.interpretation,
        }


@dataclass
class CausalConfig:
    """Configuration for causal validation."""

    pc_alpha: float = 0.05
    granger_max_lag: int = 5
    granger_test: str = "ssr_ftest"  # or 'ssr_chi2test', 'lrtest', 'params_ftest'
    partial_corr_alpha: float = 0.05


class CausalValidator:
    """
    Validates potential causal relationships.

    Methods:
    1. PC Algorithm: Discovers causal skeleton from observational data
    2. Granger Causality: Tests if X temporally precedes and predicts Y
    3. Partial Correlation: Tests if relationship holds when controlling confounds

    DISCLAIMER: These are statistical tests for causal hypotheses.
    They do not prove causation without experimental intervention.
    """

    def __init__(self, config: Optional[CausalConfig] = None):
        """Initialize causal validator."""
        self.config = config or CausalConfig()
        self.results: List[CausalResult] = []

    def pc_algorithm(
        self,
        data: np.ndarray,
        variable_names: List[str],
        alpha: Optional[float] = None,
    ) -> Dict:
        """
        Run PC algorithm to discover causal skeleton.

        The PC algorithm:
        1. Starts with fully connected graph
        2. Removes edges based on conditional independence tests
        3. Orients edges based on v-structures and rules

        Args:
            data: (n_samples, n_variables) array
            variable_names: Names for each variable
            alpha: Significance level for independence tests

        Returns:
            Dictionary with:
            - adjacency_matrix: numpy array
            - edges: list of (source, target, direction) tuples
            - graph: causal-learn graph object (if available)
        """
        if not CAUSAL_LEARN_AVAILABLE:
            logger.error("causal-learn not installed. Cannot run PC algorithm.")
            return {"error": "causal-learn not installed"}

        alpha = alpha or self.config.pc_alpha

        logger.info(
            f"Running PC algorithm on {data.shape[1]} variables "
            f"with {data.shape[0]} samples (α={alpha})"
        )

        # Run PC algorithm
        cg = pc(data, alpha=alpha, indep_test=fisherz)

        # Extract adjacency matrix
        adj_matrix = cg.G.graph

        # Parse edges with directions
        edges = []
        n_vars = len(variable_names)

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # Check edge existence and direction
                # In causal-learn: -1 means tail, 1 means arrow
                edge_ij = adj_matrix[i, j]
                edge_ji = adj_matrix[j, i]

                if edge_ij != 0 or edge_ji != 0:
                    if edge_ij == -1 and edge_ji == 1:
                        # i -> j
                        direction = "forward"
                        edges.append((variable_names[i], variable_names[j], direction))
                    elif edge_ij == 1 and edge_ji == -1:
                        # i <- j
                        direction = "backward"
                        edges.append((variable_names[j], variable_names[i], "forward"))
                    elif edge_ij == -1 and edge_ji == -1:
                        # i - j (undirected)
                        direction = "undirected"
                        edges.append((variable_names[i], variable_names[j], direction))
                    elif edge_ij == 1 and edge_ji == 1:
                        # i <-> j (bidirectional, possible latent confounder)
                        direction = "bidirectional"
                        edges.append((variable_names[i], variable_names[j], direction))

        logger.info(f"PC algorithm found {len(edges)} edges")

        return {
            "adjacency_matrix": adj_matrix,
            "edges": edges,
            "variable_names": variable_names,
            "graph": cg,
        }

    def granger_causality(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: Optional[int] = None,
        x_name: str = "X",
        y_name: str = "Y",
    ) -> CausalResult:
        """
        Test Granger causality: does X help predict Y?

        Granger causality tests whether past values of X improve
        predictions of Y beyond past values of Y alone.

        NOTE: Granger causality is about predictive causality,
        not true causality. It requires:
        - Stationarity
        - No omitted variables
        - Linear relationships

        Args:
            x: Time series of potential cause (n_timepoints,)
            y: Time series of potential effect (n_timepoints,)
            max_lag: Maximum lag to test
            x_name: Name of X variable
            y_name: Name of Y variable

        Returns:
            CausalResult with Granger test statistics
        """
        max_lag = max_lag or self.config.granger_max_lag

        # Prepare data matrix for statsmodels
        # Format: columns = [y, x], tests if x Granger-causes y
        data_matrix = np.column_stack([y, x])

        # Remove any NaN rows
        mask = ~np.any(np.isnan(data_matrix), axis=1)
        data_clean = data_matrix[mask]

        if len(data_clean) < max_lag * 3:
            logger.warning(f"Insufficient data for Granger test: {len(data_clean)} samples")
            return CausalResult(
                source=x_name,
                target=y_name,
                method="granger",
                interpretation="Insufficient data for Granger causality test",
            )

        try:
            # Run Granger causality test
            results = grangercausalitytests(
                data_clean, maxlag=max_lag, verbose=False
            )

            # Find optimal lag (lowest p-value)
            best_lag = 1
            best_p = 1.0
            best_f = 0.0

            for lag in range(1, max_lag + 1):
                test_result = results[lag][0][self.config.granger_test]
                f_stat = test_result[0]
                p_value = test_result[1]

                if p_value < best_p:
                    best_p = p_value
                    best_f = f_stat
                    best_lag = lag

            # Interpret evidence strength
            if best_p < 0.001:
                evidence = "strong"
            elif best_p < 0.01:
                evidence = "moderate"
            elif best_p < 0.05:
                evidence = "weak"
            else:
                evidence = "none"

            interpretation = (
                f"{x_name} Granger-causes {y_name} at lag {best_lag} "
                f"(F={best_f:.2f}, p={best_p:.4f}). "
                f"Evidence: {evidence}"
            )

            return CausalResult(
                source=x_name,
                target=y_name,
                method="granger",
                granger_f_stat=best_f,
                granger_p_value=best_p,
                optimal_lag=best_lag,
                causal_evidence=evidence,
                interpretation=interpretation,
            )

        except Exception as e:
            logger.error(f"Granger causality test failed: {e}")
            return CausalResult(
                source=x_name,
                target=y_name,
                method="granger",
                interpretation=f"Granger test failed: {str(e)}",
            )

    def partial_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        confounders: np.ndarray,
        x_name: str = "X",
        y_name: str = "Y",
        confounder_names: Optional[List[str]] = None,
    ) -> CausalResult:
        """
        Compute partial correlation controlling for confounders.

        Tests if the X-Y relationship remains significant
        when controlling for potential confounding variables.

        Args:
            x: Variable X (n_samples,)
            y: Variable Y (n_samples,)
            confounders: Confounding variables (n_samples, n_confounders)
            x_name: Name of X
            y_name: Name of Y
            confounder_names: Names of confounding variables

        Returns:
            CausalResult with partial correlation statistics
        """
        # Stack all variables
        if confounders.ndim == 1:
            confounders = confounders.reshape(-1, 1)

        all_data = np.column_stack([x, y, confounders])

        # Remove NaN rows
        mask = ~np.any(np.isnan(all_data), axis=1)
        data_clean = all_data[mask]

        if len(data_clean) < confounders.shape[1] + 10:
            logger.warning("Insufficient data for partial correlation")
            return CausalResult(
                source=x_name,
                target=y_name,
                method="partial_corr",
                interpretation="Insufficient data for partial correlation",
            )

        x_clean = data_clean[:, 0]
        y_clean = data_clean[:, 1]
        conf_clean = data_clean[:, 2:]

        # Compute partial correlation using regression residuals
        # Regress X on confounders, get residuals
        # Regress Y on confounders, get residuals
        # Correlate residuals

        # Add constant for intercept
        conf_with_const = np.column_stack([np.ones(len(conf_clean)), conf_clean])

        try:
            # Residualize X
            beta_x = np.linalg.lstsq(conf_with_const, x_clean, rcond=None)[0]
            resid_x = x_clean - conf_with_const @ beta_x

            # Residualize Y
            beta_y = np.linalg.lstsq(conf_with_const, y_clean, rcond=None)[0]
            resid_y = y_clean - conf_with_const @ beta_y

            # Partial correlation = correlation of residuals
            partial_r, p_value = stats.pearsonr(resid_x, resid_y)

            # Interpret
            if p_value < self.config.partial_corr_alpha and abs(partial_r) > 0.1:
                evidence = "moderate" if abs(partial_r) > 0.3 else "weak"
                interp = (
                    f"Relationship between {x_name} and {y_name} remains significant "
                    f"after controlling for confounders (r={partial_r:.3f}, p={p_value:.4f})"
                )
            else:
                evidence = "none"
                interp = (
                    f"Relationship between {x_name} and {y_name} is not significant "
                    f"after controlling for confounders (r={partial_r:.3f}, p={p_value:.4f})"
                )

            return CausalResult(
                source=x_name,
                target=y_name,
                method="partial_corr",
                partial_corr=partial_r,
                partial_p_value=p_value,
                controlled_vars=confounder_names,
                causal_evidence=evidence,
                interpretation=interp,
            )

        except Exception as e:
            logger.error(f"Partial correlation failed: {e}")
            return CausalResult(
                source=x_name,
                target=y_name,
                method="partial_corr",
                interpretation=f"Partial correlation failed: {str(e)}",
            )

    def validate_relationship(
        self,
        sensor_data: np.ndarray,
        survey_score: np.ndarray,
        sensor_name: str,
        survey_name: str,
        confounders: Optional[np.ndarray] = None,
        confounder_names: Optional[List[str]] = None,
    ) -> Dict[str, CausalResult]:
        """
        Comprehensive causal validation of a single relationship.

        Runs multiple validation methods:
        1. Granger causality (if temporal data)
        2. Partial correlation (if confounders provided)

        Args:
            sensor_data: Sensor feature time series
            survey_score: Survey score (can be single value or time series)
            sensor_name: Name of sensor feature
            survey_name: Name of survey item
            confounders: Optional confounding variables
            confounder_names: Names of confounders

        Returns:
            Dictionary of method -> CausalResult
        """
        results = {}

        # Granger causality (requires time series)
        if len(sensor_data.shape) == 1 and len(sensor_data) > 20:
            # If survey is single value, can't do Granger
            if len(survey_score.shape) == 1 and len(survey_score) == len(sensor_data):
                results["granger"] = self.granger_causality(
                    sensor_data, survey_score, x_name=sensor_name, y_name=survey_name
                )

        # Partial correlation
        if confounders is not None:
            results["partial_corr"] = self.partial_correlation(
                sensor_data,
                survey_score,
                confounders,
                x_name=sensor_name,
                y_name=survey_name,
                confounder_names=confounder_names,
            )

        return results

    def summarize_evidence(self, results: Dict[str, CausalResult]) -> str:
        """
        Summarize causal evidence from multiple methods.

        Args:
            results: Dictionary of method -> CausalResult

        Returns:
            Overall evidence summary
        """
        evidence_levels = {"strong": 3, "moderate": 2, "weak": 1, "none": 0}

        max_evidence = "none"
        max_level = 0

        for method, result in results.items():
            level = evidence_levels.get(result.causal_evidence, 0)
            if level > max_level:
                max_level = level
                max_evidence = result.causal_evidence

        return max_evidence
