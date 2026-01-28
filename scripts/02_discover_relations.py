#!/usr/bin/env python3
"""
Step 2: Relationship Discovery

Discovers statistical relationships between sensor features and survey scores.
Supports both:
1. User-level analysis (aggregated features vs survey scores)
2. Temporal analysis (sliding window features vs survey scores)

Applies multiple comparison correction and effect size filtering.

Usage:
    python scripts/02_discover_relations.py

Outputs:
    - outputs/relations/all_correlations.csv
    - outputs/relations/significant_relations.json
    - outputs/relations/relation_heatmap.png
    - outputs/relations/temporal_correlations.csv
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.discovery.correlation import RelationshipDiscovery, DiscoveryConfig


def setup_logging(output_dir: Path) -> None:
    """Configure logging."""
    log_file = output_dir / "logs" / f"discover_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, rotation="10 MB", level="DEBUG")


def load_data(processed_dir: Path) -> tuple:
    """Load preprocessed data."""
    # Load user-level sensor features
    features_path = processed_dir / "sensor_features.parquet"
    features_df = pl.read_parquet(features_path)
    logger.info(f"Loaded user-level features: {features_df.shape}")

    # Load temporal features (sliding window)
    temporal_path = processed_dir / "temporal_features.parquet"
    temporal_df = None
    if temporal_path.exists():
        temporal_df = pl.read_parquet(temporal_path)
        logger.info(f"Loaded temporal features: {temporal_df.shape}")

    # Load survey scores
    survey_path = processed_dir / "survey_scores.parquet"
    survey_df = pl.read_parquet(survey_path)
    logger.info(f"Loaded survey scores: {survey_df.shape}")

    # Load grades if available
    grades_path = processed_dir / "grades.parquet"
    grades_df = None
    if grades_path.exists():
        grades_df = pl.read_parquet(grades_path)
        logger.info(f"Loaded grades: {grades_df.shape}")

    return features_df, temporal_df, survey_df, grades_df


def prepare_feature_matrix(features_df: pl.DataFrame) -> tuple:
    """Extract numeric feature matrix from user-level features."""
    # Get numeric columns (exclude user_id and validity flags)
    exclude_cols = ["user_id", "gps_valid", "activity_valid", "phone_valid", "conv_valid"]
    feature_cols = [
        c for c in features_df.columns
        if c not in exclude_cols and features_df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
    ]

    # Extract matrix
    feature_matrix = features_df.select(feature_cols).to_numpy()
    user_ids = features_df["user_id"].to_list()

    logger.info(f"User-level feature matrix shape: {feature_matrix.shape}")
    logger.info(f"Features: {feature_cols}")

    return feature_matrix, feature_cols, user_ids


def prepare_temporal_feature_matrix(temporal_df: pl.DataFrame, survey_df: pl.DataFrame) -> tuple:
    """
    Prepare temporal feature matrix aligned with pre-survey timing.

    Strategy: For each user, aggregate temporal features (mean over study period).
    """
    if temporal_df is None:
        return None, None, None

    # Get feature columns (exclude user_id and date)
    # Only include numeric columns
    feature_cols = [
        c for c in temporal_df.columns
        if c not in ["user_id", "date"]
        and temporal_df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
    ]

    if not feature_cols:
        logger.warning("No numeric temporal features found")
        return None, None, None

    # Aggregate all temporal features per user (mean)
    user_temporal = temporal_df.group_by("user_id").agg([
        pl.col(c).mean().alias(c) for c in feature_cols
    ])

    # Filter out None user_ids
    user_temporal = user_temporal.filter(pl.col("user_id").is_not_null())

    user_ids = user_temporal["user_id"].to_list()
    feature_matrix = user_temporal.select(feature_cols).to_numpy().astype(np.float64)

    logger.info(f"Temporal feature matrix shape: {feature_matrix.shape}")
    logger.info(f"Temporal features: {len(feature_cols)} columns")

    return feature_matrix, feature_cols, user_ids


def prepare_survey_matrix(survey_df: pl.DataFrame, user_ids: list) -> tuple:
    """Extract survey score matrix aligned with feature users."""
    # Use pre-test scores only (to avoid temporal leakage)
    pre_df = survey_df.filter(pl.col("type") == "pre")

    # Get score columns
    score_cols = ["total_score", "Q1_score", "Q2_score", "Q3_score", "Q4_score",
                  "Q5_score", "Q6_score", "Q7_score", "Q8_score", "Q9_score"]
    available_cols = [c for c in score_cols if c in pre_df.columns]

    if not available_cols:
        logger.warning("No score columns found in survey data")
        return None, None

    # Create user-aligned matrix
    survey_matrix = []
    for uid in user_ids:
        user_data = pre_df.filter(pl.col("uid") == uid)
        if len(user_data) > 0:
            row = user_data.select(available_cols).to_numpy()[0]
        else:
            row = [np.nan] * len(available_cols)
        survey_matrix.append(row)

    survey_matrix = np.array(survey_matrix, dtype=float)
    logger.info(f"Survey matrix shape: {survey_matrix.shape}")
    logger.info(f"Survey columns: {available_cols}")

    return survey_matrix, available_cols


def prepare_grades_matrix(grades_df: pl.DataFrame, user_ids: list) -> tuple:
    """Extract grades matrix aligned with feature users."""
    if grades_df is None:
        return None, None

    # Get GPA columns and clean names
    gpa_cols = [c for c in grades_df.columns if "gpa" in c.lower()]

    if not gpa_cols:
        return None, None

    # Clean column names (remove spaces, standardize)
    clean_names = [c.strip().replace(" ", "_") for c in gpa_cols]

    # Create user-aligned matrix
    grades_matrix = []
    for uid in user_ids:
        user_data = grades_df.filter(pl.col("uid") == uid)
        if len(user_data) > 0:
            row = user_data.select(gpa_cols).to_numpy()[0]
        else:
            row = [np.nan] * len(gpa_cols)
        grades_matrix.append(row)

    grades_matrix = np.array(grades_matrix, dtype=float)
    logger.info(f"Grades matrix shape: {grades_matrix.shape}")
    logger.info(f"GPA columns: {clean_names}")

    return grades_matrix, clean_names


def plot_correlation_heatmap(
    results: list,
    feature_names: list,
    survey_names: list,
    output_path: Path,
    title: str = "Sensor-Survey Correlations"
) -> None:
    """Create correlation heatmap visualization."""
    # Build correlation matrix
    n_features = len(feature_names)
    n_surveys = len(survey_names)
    corr_matrix = np.zeros((n_features, n_surveys))
    pval_matrix = np.ones((n_features, n_surveys))

    for r in results:
        if r.sensor_feature in feature_names and r.survey_item in survey_names:
            i = feature_names.index(r.sensor_feature)
            j = survey_names.index(r.survey_item)
            corr_matrix[i, j] = r.correlation
            pval_matrix[i, j] = r.p_adjusted

    # Limit features shown for readability
    if n_features > 30:
        # Show only features with at least one significant correlation
        sig_features = []
        for i, f in enumerate(feature_names):
            if np.any(pval_matrix[i, :] < 0.1):  # Use 0.1 threshold for display
                sig_features.append((i, f))

        if sig_features:
            indices = [x[0] for x in sig_features]
            feature_names = [x[1] for x in sig_features]
            corr_matrix = corr_matrix[indices, :]
            pval_matrix = pval_matrix[indices, :]
            n_features = len(feature_names)

    if n_features == 0:
        logger.warning("No features to plot in heatmap")
        return

    # Create figure
    fig_height = max(6, n_features * 0.3)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Heatmap
    sns.heatmap(
        corr_matrix,
        xticklabels=survey_names,
        yticklabels=feature_names,
        cmap="RdBu_r",
        center=0,
        vmin=-0.6,
        vmax=0.6,
        annot=True if n_features <= 20 else False,
        fmt=".2f" if n_features <= 20 else "",
        ax=ax,
    )

    # Mark significant cells
    for i in range(n_features):
        for j in range(n_surveys):
            if pval_matrix[i, j] < 0.05:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=False, edgecolor='gold', linewidth=2
                ))

    ax.set_title(f"{title}\n(gold border = p_adj < 0.05)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved heatmap to {output_path}")


def convert_to_json_serializable(obj):
    """Convert numpy types to JSON serializable Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_results(
    results: list,
    significant: list,
    output_dir: Path,
    prefix: str = ""
) -> None:
    """Save results to files."""
    prefix_str = f"{prefix}_" if prefix else ""

    # All correlations as CSV
    all_corr_path = output_dir / f"{prefix_str}all_correlations.csv"
    rows = [r.to_dict() for r in results]
    pl.DataFrame(rows).write_csv(all_corr_path)
    logger.info(f"Saved all correlations to {all_corr_path}")

    # Significant relations as JSON
    sig_path = output_dir / f"{prefix_str}significant_relations.json"
    sig_data = {
        "n_total": len(results),
        "n_significant": len(significant),
        "alpha": 0.05,
        "min_effect_size": 0.3,
        "relations": [convert_to_json_serializable(r.to_dict()) for r in significant]
    }
    with open(sig_path, "w") as f:
        json.dump(sig_data, f, indent=2)
    logger.info(f"Saved significant relations to {sig_path}")


def main():
    parser = argparse.ArgumentParser(description="Discover sensor-survey relationships")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/processed",
        help="Input directory with preprocessed data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/relations",
        help="Output directory",
    )
    parser.add_argument(
        "--correction",
        type=str,
        default="fdr_bh",
        choices=["fdr_bh", "bonferroni", "holm", "none"],
        help="Multiple comparison correction method",
    )
    parser.add_argument(
        "--min-effect-size",
        type=float,
        default=0.3,
        help="Minimum effect size (Cohen's d equivalent)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="spearman",
        choices=["pearson", "spearman"],
        help="Correlation method",
    )
    parser.add_argument(
        "--analyze-temporal",
        action="store_true",
        default=True,
        help="Also analyze temporal (sliding window) features",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    np.random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("Relationship Discovery")
    logger.info("=" * 60)
    logger.info(f"Correction method: {args.correction}")
    logger.info(f"Min effect size: {args.min_effect_size}")
    logger.info(f"Correlation method: {args.method}")

    # Load data
    processed_dir = Path(args.input)
    features_df, temporal_df, survey_df, grades_df = load_data(processed_dir)

    # Configure discovery
    config = DiscoveryConfig(
        method=args.method,
        correction=args.correction,
        alpha=0.05,
        min_effect_size=args.min_effect_size,
        min_samples=20,
        bootstrap_ci=False,
    )

    discovery = RelationshipDiscovery(config)
    all_results = []
    all_significant = []

    # ==========================================================================
    # Part 1: User-Level Analysis
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PART 1: User-Level Analysis")
    logger.info("=" * 60)

    # Prepare matrices
    feature_matrix, feature_names, user_ids = prepare_feature_matrix(features_df)
    survey_matrix, survey_names = prepare_survey_matrix(survey_df, user_ids)

    if survey_matrix is not None:
        logger.info("\n--- Analyzing User-Level Sensor-PHQ9 Relationships ---")
        results = discovery.compute_all_correlations(
            feature_matrix, survey_matrix,
            feature_names, survey_names
        )
        all_results.extend(results)

        # Get significant results
        significant = discovery.filter_significant()
        all_significant.extend(significant)

        # Summary
        stats = discovery.get_summary_stats()
        logger.info(f"Total tests: {stats.get('n_tests', 0)}")
        logger.info(f"Significant (p_adj < 0.05): {stats.get('n_significant', 0)}")
        logger.info(f"Meaningful (effect >= {args.min_effect_size}): {stats.get('n_meaningful', 0)}")

        # Visualization
        plot_correlation_heatmap(
            results, feature_names, survey_names,
            output_dir / "user_level_heatmap.png",
            "User-Level Sensor-Survey Correlations"
        )

    # Add grades analysis if available
    grades_matrix, gpa_cols = prepare_grades_matrix(grades_df, user_ids)
    if grades_matrix is not None:
        logger.info("\n--- Analyzing User-Level Sensor-GPA Relationships ---")
        discovery_grades = RelationshipDiscovery(config)
        gpa_results = discovery_grades.compute_all_correlations(
            feature_matrix, grades_matrix,
            feature_names, gpa_cols
        )
        all_results.extend(gpa_results)
        all_significant.extend(discovery_grades.filter_significant())

    # ==========================================================================
    # Part 2: Temporal Feature Analysis
    # ==========================================================================
    if args.analyze_temporal and temporal_df is not None:
        logger.info("\n" + "=" * 60)
        logger.info("PART 2: Temporal Feature Analysis")
        logger.info("=" * 60)

        # Prepare temporal feature matrix
        temporal_matrix, temporal_cols, temporal_user_ids = prepare_temporal_feature_matrix(
            temporal_df, survey_df
        )

        if temporal_matrix is not None and len(temporal_matrix) > 0:
            # Prepare survey matrix aligned with temporal users
            survey_matrix_t, survey_names_t = prepare_survey_matrix(survey_df, temporal_user_ids)

            if survey_matrix_t is not None:
                logger.info("\n--- Analyzing Temporal Sensor-PHQ9 Relationships ---")
                discovery_temporal = RelationshipDiscovery(config)
                temporal_results = discovery_temporal.compute_all_correlations(
                    temporal_matrix, survey_matrix_t,
                    temporal_cols, survey_names_t
                )

                temporal_significant = discovery_temporal.filter_significant()

                # Summary
                stats_t = discovery_temporal.get_summary_stats()
                logger.info(f"Total tests: {stats_t.get('n_tests', 0)}")
                logger.info(f"Significant (p_adj < 0.05): {stats_t.get('n_significant', 0)}")
                logger.info(f"Meaningful (effect >= {args.min_effect_size}): {stats_t.get('n_meaningful', 0)}")

                # Save temporal results separately
                save_results(
                    temporal_results, temporal_significant,
                    output_dir, prefix="temporal"
                )

                # Add to all results
                all_results.extend(temporal_results)
                all_significant.extend(temporal_significant)

                # Visualization (top features only due to large number)
                if temporal_significant:
                    # Get top features by effect size
                    top_features = sorted(
                        set(r.sensor_feature for r in temporal_significant),
                        key=lambda f: max(
                            abs(r.correlation) for r in temporal_significant
                            if r.sensor_feature == f
                        ),
                        reverse=True
                    )[:20]

                    plot_correlation_heatmap(
                        [r for r in temporal_results if r.sensor_feature in top_features],
                        top_features, survey_names_t,
                        output_dir / "temporal_heatmap.png",
                        "Top Temporal Feature-Survey Correlations"
                    )

    # ==========================================================================
    # Final Summary and Save
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total correlations computed: {len(all_results)}")
    logger.info(f"Total significant relations: {len(all_significant)}")

    if all_significant:
        logger.info("\n--- Top Significant Relationships ---")
        sorted_sig = sorted(all_significant, key=lambda x: abs(x.correlation), reverse=True)
        for r in sorted_sig[:15]:
            logger.info(
                f"  {r.sensor_feature} <-> {r.survey_item}: "
                f"r={r.correlation:.3f}, p_adj={r.p_adjusted:.4f}, "
                f"effect={r.effect_size:.3f} ({r.interpretation})"
            )

    # Save all results
    save_results(all_results, all_significant, output_dir)

    logger.info("\nRelationship discovery complete!")


if __name__ == "__main__":
    main()
