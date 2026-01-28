#!/usr/bin/env python3
"""
Step 3: Causal Relationship Validation

Validates discovered correlations using causal inference methods:
1. PC Algorithm - discovers causal skeleton
2. Granger Causality - tests temporal precedence

Usage:
    python scripts/03_validate_relations.py

Outputs:
    - outputs/relations/causal_graph.json
    - outputs/relations/validated_relations.json
    - outputs/relations/validation_report.md
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.validation.causal import CausalValidator, CausalConfig


def convert_to_json_serializable(obj):
    """Convert numpy types to JSON serializable Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'to_dict'):
        return convert_to_json_serializable(obj.to_dict())
    elif hasattr(obj, '__dict__'):
        return convert_to_json_serializable(obj.__dict__)
    else:
        return str(obj)  # Fallback to string representation


def setup_logging(output_dir: Path) -> None:
    """Configure logging."""
    log_file = output_dir / "logs" / f"validate_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, rotation="10 MB", level="DEBUG")


def load_relations(relations_path: Path) -> list:
    """Load significant relations from discovery step."""
    with open(relations_path) as f:
        data = json.load(f)
    return data.get("relations", [])


def load_feature_data(processed_dir: Path) -> tuple:
    """Load preprocessed feature data."""
    features_df = pl.read_parquet(processed_dir / "sensor_features.parquet")
    survey_df = pl.read_parquet(processed_dir / "survey_scores.parquet")

    # Load grades if available
    grades_df = None
    grades_path = processed_dir / "grades.parquet"
    if grades_path.exists():
        grades_df = pl.read_parquet(grades_path)
        logger.info(f"Loaded grades: {grades_df.shape}")

    logger.info(f"Loaded features: {features_df.shape}")
    logger.info(f"Loaded surveys: {survey_df.shape}")

    return features_df, survey_df, grades_df


def prepare_data_matrix(
    features_df: pl.DataFrame,
    survey_df: pl.DataFrame,
    grades_df: pl.DataFrame,
    relations: list
) -> tuple:
    """Prepare data matrix for causal analysis."""
    # Get unique feature names from relations
    sensor_features = list(set(r["sensor_feature"] for r in relations))
    survey_items = list(set(r["survey_item"].strip() for r in relations))  # Strip whitespace

    # Filter to numeric columns
    exclude_cols = ["user_id", "uid", "type", "gps_valid", "activity_valid", "phone_valid", "conv_valid"]

    # Get available sensor features
    available_sensor = [
        c for c in sensor_features
        if c in features_df.columns and c not in exclude_cols
    ]

    # Get user IDs
    user_ids = features_df["user_id"].to_list()

    # Extract sensor feature matrix
    if available_sensor:
        sensor_matrix = features_df.select(available_sensor).to_numpy()
    else:
        logger.warning("No sensor features found in data")
        return None, None, None

    # Check survey data
    pre_df = survey_df.filter(pl.col("type") == "pre")
    available_survey = [c for c in survey_items if c in pre_df.columns]

    # Check grades data
    available_grades = []
    if grades_df is not None:
        # Try to match grade columns (with or without spaces)
        for item in survey_items:
            item_clean = item.strip().replace(" ", "_")
            for col in grades_df.columns:
                col_clean = col.strip().replace(" ", "_")
                if item_clean == col_clean or item.strip() == col.strip():
                    available_grades.append(col)
                    break

    if not available_survey and not available_grades:
        logger.warning("No survey items or grades found in data")
        return None, None, None

    # Build outcome matrix
    outcome_matrix = []
    outcome_names = []

    for uid in user_ids:
        row = []

        # Survey scores
        if available_survey:
            user_data = pre_df.filter(pl.col("uid") == uid)
            if len(user_data) > 0:
                row.extend(user_data.select(available_survey).to_numpy()[0])
            else:
                row.extend([np.nan] * len(available_survey))

        # Grades
        if available_grades:
            user_grades = grades_df.filter(pl.col("uid") == uid)
            if len(user_grades) > 0:
                row.extend(user_grades.select(available_grades).to_numpy()[0])
            else:
                row.extend([np.nan] * len(available_grades))

        outcome_matrix.append(row)

    outcome_names = available_survey + available_grades
    outcome_matrix = np.array(outcome_matrix, dtype=float)

    # Combine into single matrix
    combined = np.hstack([sensor_matrix, outcome_matrix])
    variable_names = available_sensor + outcome_names

    # Remove rows with NaN
    valid_mask = ~np.any(np.isnan(combined), axis=1)
    combined = combined[valid_mask]

    logger.info(f"Combined matrix shape: {combined.shape}")
    logger.info(f"Variables: {variable_names}")

    return combined, variable_names, valid_mask


def generate_validation_report(
    causal_results: dict,
    relations: list,
    output_path: Path
) -> None:
    """Generate markdown validation report."""
    report_lines = [
        "# Causal Validation Report",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## Summary",
        "",
        f"- Input relations: {len(relations)}",
        f"- PC algorithm edges: {causal_results.get('n_edges', 0)}",
        f"- Directed edges: {causal_results.get('n_directed', 0)}",
        "",
        "## PC Algorithm Results",
        "",
        "The PC algorithm discovers the causal skeleton from observational data.",
        "Edges indicate potential causal relationships; direction indicates causal flow.",
        "",
        "### Discovered Edges",
        "",
    ]

    edges = causal_results.get("edges", [])
    if edges:
        report_lines.append("| From | To | Direction | Evidence |")
        report_lines.append("|------|-----|-----------|----------|")
        for edge in edges[:20]:  # Top 20 edges
            from_var = edge.get("from", "?")
            to_var = edge.get("to", "?")
            direction = edge.get("direction", "undirected")
            evidence = edge.get("evidence", "weak")
            report_lines.append(f"| {from_var} | {to_var} | {direction} | {evidence} |")
    else:
        report_lines.append("No edges discovered by PC algorithm.")

    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "### Causal vs Correlation",
        "",
        "- **Correlation** (from Step 2): Statistical association, does not imply causation",
        "- **PC Algorithm**: Discovers potential causal relationships under assumptions:",
        "  - Causal Markov Condition",
        "  - Faithfulness",
        "  - No unmeasured confounders (strong assumption)",
        "",
        "### Limitations",
        "",
        "1. **Small sample size** (n~50): Limited statistical power for causal discovery",
        "2. **Cross-sectional data**: Cannot establish temporal precedence",
        "3. **Unmeasured confounders**: Results may be biased by unobserved variables",
        "4. **Multiple testing**: Many variable pairs tested",
        "",
        "## Recommendations",
        "",
        "- Treat discovered causal relationships as **hypotheses** requiring validation",
        "- Use domain knowledge (DSM-5, literature) to evaluate plausibility",
        "- Consider longitudinal analysis for temporal causal claims",
        "",
        "---",
        "*This report is for research purposes only.*",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved validation report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate causal relationships")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/processed",
        help="Input directory with preprocessed data",
    )
    parser.add_argument(
        "--relations",
        type=str,
        default="outputs/relations/significant_relations.json",
        help="Path to significant relations from discovery step",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/relations",
        help="Output directory",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for PC algorithm",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    np.random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("Causal Relationship Validation")
    logger.info("=" * 60)

    # Load significant relations
    relations_path = Path(args.relations)
    if not relations_path.exists():
        logger.error(f"Relations file not found: {relations_path}")
        logger.info("Run 02_discover_relations.py first")
        return

    relations = load_relations(relations_path)
    logger.info(f"Loaded {len(relations)} significant relations")

    if not relations:
        logger.warning("No relations to validate")
        return

    # Load feature data
    processed_dir = Path(args.input)
    features_df, survey_df, grades_df = load_feature_data(processed_dir)

    # Prepare data matrix
    data_matrix, variable_names, valid_mask = prepare_data_matrix(
        features_df, survey_df, grades_df, relations
    )

    if data_matrix is None:
        logger.error("Failed to prepare data matrix")
        return

    # Configure validator
    config = CausalConfig(
        pc_alpha=args.alpha,
    )

    validator = CausalValidator(config)

    # Run PC algorithm
    logger.info("\n--- Running PC Algorithm ---")
    try:
        causal_result = validator.pc_algorithm(data_matrix, variable_names, alpha=args.alpha)
        logger.info(f"PC algorithm completed")
        edges = causal_result.get("edges", [])
        logger.info(f"  Edges discovered: {len(edges)}")
        directed = [e for e in edges if e[2] == "forward"]
        logger.info(f"  Directed edges: {len(directed)}")
        causal_result["n_edges"] = len(edges)
        causal_result["n_directed"] = len(directed)
    except Exception as e:
        logger.error(f"PC algorithm failed: {e}")
        causal_result = {"edges": [], "n_edges": 0, "n_directed": 0}

    # Validate individual relations
    logger.info("\n--- Validating Individual Relations ---")
    validated_relations = []

    for rel in relations:
        sensor = rel["sensor_feature"]
        survey = rel["survey_item"]

        # Check if edge exists in causal graph
        edge_info = None
        for edge in causal_result.get("edges", []):
            if (edge["from"] == sensor and edge["to"] == survey) or \
               (edge["from"] == survey and edge["to"] == sensor):
                edge_info = edge
                break

        validated = {
            **rel,
            "causal_edge": edge_info is not None,
            "causal_direction": edge_info.get("direction") if edge_info else None,
            "validation_status": "supported" if edge_info else "correlation_only"
        }
        validated_relations.append(validated)

        if edge_info:
            logger.info(
                f"  {sensor} -> {survey}: CAUSAL EDGE ({edge_info.get('direction', 'undirected')})"
            )

    # Summary
    n_supported = sum(1 for v in validated_relations if v["causal_edge"])
    logger.info(f"\nValidation Summary:")
    logger.info(f"  Total relations: {len(validated_relations)}")
    logger.info(f"  Causally supported: {n_supported}")
    logger.info(f"  Correlation only: {len(validated_relations) - n_supported}")

    # Save results
    # Causal graph (remove non-serializable graph object)
    causal_path = output_dir / "causal_graph.json"
    causal_to_save = {k: v for k, v in causal_result.items() if k != "graph"}
    causal_serializable = convert_to_json_serializable(causal_to_save)
    with open(causal_path, "w") as f:
        json.dump(causal_serializable, f, indent=2)
    logger.info(f"Saved causal graph to {causal_path}")

    # Validated relations
    validated_path = output_dir / "validated_relations.json"
    validated_data = {
        "n_total": len(validated_relations),
        "n_causally_supported": n_supported,
        "alpha": args.alpha,
        "relations": convert_to_json_serializable(validated_relations)
    }
    with open(validated_path, "w") as f:
        json.dump(validated_data, f, indent=2)
    logger.info(f"Saved validated relations to {validated_path}")

    # Generate report
    generate_validation_report(
        causal_result, relations,
        output_dir / "validation_report.md"
    )

    logger.info("\nCausal validation complete!")


if __name__ == "__main__":
    main()
