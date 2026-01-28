#!/usr/bin/env python3
"""
Step 4: Knowledge Graph Construction

Builds a heterogeneous knowledge graph combining:
1. Discovered statistical relationships
2. DSM-5 clinical knowledge
3. User feature nodes

Usage:
    python scripts/04_build_kg.py

Outputs:
    - outputs/graphs/knowledge_graph.json
    - outputs/graphs/kg_visualization.png
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl
import networkx as nx
import matplotlib.pyplot as plt
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge.schemas.dsm5 import (
    MDDSymptom,
    PHQ9Item,
    DIGITAL_BEHAVIORS,
    DSM5_BEHAVIOR_MAPPINGS,
    PHQ9_TO_DSM5
)


def setup_logging(output_dir: Path) -> None:
    """Configure logging."""
    log_file = output_dir / "logs" / f"build_kg_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, rotation="10 MB", level="DEBUG")


def load_validated_relations(relations_path: Path) -> list:
    """Load validated relations."""
    with open(relations_path) as f:
        data = json.load(f)
    return data.get("relations", [])


def create_dsm5_subgraph() -> nx.DiGraph:
    """Create DSM-5 knowledge subgraph."""
    G = nx.DiGraph()

    # Add MDD node
    G.add_node(
        "MDD",
        node_type="disorder",
        name="Major Depressive Disorder",
        source="DSM-5"
    )

    # Add symptom nodes
    for symptom in MDDSymptom:
        G.add_node(
            symptom.value,
            node_type="symptom",
            name=symptom.name.replace("_", " ").title(),
            source="DSM-5"
        )
        # Link to MDD
        G.add_edge("MDD", symptom.value, edge_type="has_symptom")

    # Add PHQ-9 item nodes and link to symptoms
    for item, symptom in PHQ9_TO_DSM5.items():
        item_id = f"PHQ9_{item.name}"
        G.add_node(
            item_id,
            node_type="assessment_item",
            name=item.value,
            source="PHQ-9"
        )
        G.add_edge(item_id, symptom.value, edge_type="measures")

    # Add digital behavior nodes and link to symptoms
    for mapping in DSM5_BEHAVIOR_MAPPINGS:
        symptom_id = mapping.symptom.value
        for behavior_name in mapping.digital_behaviors:
            if behavior_name in DIGITAL_BEHAVIORS:
                behavior = DIGITAL_BEHAVIORS[behavior_name]
                if not G.has_node(behavior_name):
                    G.add_node(
                        behavior_name,
                        node_type="digital_behavior",
                        name=behavior.name,
                        description=behavior.description,
                        source="literature"
                    )
                G.add_edge(
                    behavior_name,
                    symptom_id,
                    edge_type="indicates",
                    evidence_level=mapping.evidence_level.value,
                    references=mapping.references
                )

    return G


def add_statistical_relations(
    G: nx.DiGraph,
    relations: list,
    min_effect_size: float = 0.2
) -> nx.DiGraph:
    """Add discovered statistical relationships to graph."""
    for rel in relations:
        sensor = rel["sensor_feature"]
        survey = rel["survey_item"]
        correlation = rel.get("correlation", 0)
        effect_size = rel.get("effect_size", 0)
        p_adjusted = rel.get("p_adjusted", 1)

        # Skip weak effects
        if abs(effect_size) < min_effect_size:
            continue

        # Ensure sensor node exists
        if not G.has_node(sensor):
            G.add_node(
                sensor,
                node_type="sensor_feature",
                name=sensor,
                source="studentlife"
            )

        # Ensure survey node exists
        if not G.has_node(survey):
            G.add_node(
                survey,
                node_type="survey_item",
                name=survey,
                source="studentlife"
            )

        # Add edge with statistical properties
        G.add_edge(
            sensor,
            survey,
            edge_type="correlates_with",
            correlation=correlation,
            effect_size=effect_size,
            p_adjusted=p_adjusted,
            causal_supported=rel.get("causal_edge", False),
            source="analysis"
        )

    return G


def add_user_nodes(
    G: nx.DiGraph,
    features_df: pl.DataFrame,
    survey_df: pl.DataFrame
) -> nx.DiGraph:
    """Add user nodes with their feature values."""
    user_ids = features_df["user_id"].to_list()
    pre_df = survey_df.filter(pl.col("type") == "pre")

    for uid in user_ids:
        # Get user features
        user_features = features_df.filter(pl.col("user_id") == uid)
        if len(user_features) == 0:
            continue

        # Get user survey scores
        user_survey = pre_df.filter(pl.col("uid") == uid)

        user_data = {
            "node_type": "user",
            "source": "studentlife"
        }

        # Add feature values
        for col in features_df.columns:
            if col not in ["user_id", "gps_valid", "activity_valid", "phone_valid", "conv_valid"]:
                val = user_features[col].to_list()[0]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    user_data[f"feature_{col}"] = float(val)

        # Add survey scores
        if len(user_survey) > 0:
            for col in user_survey.columns:
                if col not in ["uid", "type"]:
                    val = user_survey[col].to_list()[0]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        user_data[f"survey_{col}"] = float(val)

        G.add_node(uid, **user_data)

        # Link user to their active sensor features
        for col in features_df.columns:
            if col in ["user_id", "gps_valid", "activity_valid", "phone_valid", "conv_valid"]:
                continue
            if G.has_node(col):
                val = user_features[col].to_list()[0]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    G.add_edge(
                        uid,
                        col,
                        edge_type="has_value",
                        value=float(val)
                    )

    return G


def visualize_knowledge_graph(G: nx.DiGraph, output_path: Path) -> None:
    """Create knowledge graph visualization."""
    fig, ax = plt.subplots(figsize=(16, 12))

    # Color by node type
    color_map = {
        "disorder": "#e74c3c",
        "symptom": "#f39c12",
        "assessment_item": "#3498db",
        "digital_behavior": "#2ecc71",
        "sensor_feature": "#9b59b6",
        "survey_item": "#1abc9c",
        "user": "#95a5a6"
    }

    # Filter out user nodes for cleaner visualization
    viz_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") != "user"]
    subgraph = G.subgraph(viz_nodes)

    node_colors = [
        color_map.get(subgraph.nodes[n].get("node_type", ""), "#cccccc")
        for n in subgraph.nodes()
    ]

    # Layout
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(
        subgraph, pos,
        node_color=node_colors,
        node_size=500,
        alpha=0.8,
        ax=ax
    )

    # Draw edges by type
    edge_colors = {
        "has_symptom": "#e74c3c",
        "measures": "#3498db",
        "indicates": "#2ecc71",
        "correlates_with": "#9b59b6"
    }

    for edge_type, color in edge_colors.items():
        edges = [(u, v) for u, v, d in subgraph.edges(data=True)
                 if d.get("edge_type") == edge_type]
        if edges:
            nx.draw_networkx_edges(
                subgraph, pos,
                edgelist=edges,
                edge_color=color,
                alpha=0.5,
                arrows=True,
                arrowsize=10,
                ax=ax
            )

    # Draw labels
    labels = {n: n[:15] + "..." if len(n) > 15 else n for n in subgraph.nodes()}
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, ax=ax)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                   markersize=10, label=node_type.replace("_", " ").title())
        for node_type, color in color_map.items()
        if node_type != "user"
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    ax.set_title("Mental Health Knowledge Graph\n(excluding user nodes)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved visualization to {output_path}")


def export_graph(G: nx.DiGraph, output_path: Path) -> None:
    """Export graph to JSON format."""
    # Convert to serializable format
    data = {
        "nodes": [],
        "edges": [],
        "metadata": {
            "created": datetime.now().isoformat(),
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges()
        }
    }

    # Export nodes
    for node_id, attrs in G.nodes(data=True):
        node_data = {"id": node_id, **attrs}
        # Convert non-serializable values
        for k, v in node_data.items():
            if isinstance(v, (np.floating, np.integer)):
                node_data[k] = float(v)
        data["nodes"].append(node_data)

    # Export edges
    for u, v, attrs in G.edges(data=True):
        edge_data = {"source": u, "target": v, **attrs}
        # Convert non-serializable values
        for k, v in edge_data.items():
            if isinstance(v, (np.floating, np.integer)):
                edge_data[k] = float(v)
            elif isinstance(v, list):
                edge_data[k] = [str(x) for x in v]
        data["edges"].append(edge_data)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Exported graph to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build knowledge graph")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/processed",
        help="Input directory with preprocessed data",
    )
    parser.add_argument(
        "--relations",
        type=str,
        default="outputs/relations/validated_relations.json",
        help="Path to validated relations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/graphs",
        help="Output directory",
    )
    parser.add_argument(
        "--min-effect-size",
        type=float,
        default=0.2,
        help="Minimum effect size for including relations",
    )
    parser.add_argument(
        "--include-users",
        action="store_true",
        help="Include user nodes in graph",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    np.random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("Knowledge Graph Construction")
    logger.info("=" * 60)

    # Create DSM-5 subgraph
    logger.info("\n--- Creating DSM-5 Knowledge Subgraph ---")
    G = create_dsm5_subgraph()
    logger.info(f"DSM-5 subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load and add statistical relations
    relations_path = Path(args.relations)
    if relations_path.exists():
        logger.info("\n--- Adding Statistical Relations ---")
        relations = load_validated_relations(relations_path)
        logger.info(f"Loaded {len(relations)} validated relations")
        G = add_statistical_relations(G, relations, args.min_effect_size)
        logger.info(f"After adding relations: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        logger.warning(f"Relations file not found: {relations_path}")
        logger.info("Using DSM-5 knowledge only")

    # Add user nodes if requested
    if args.include_users:
        logger.info("\n--- Adding User Nodes ---")
        processed_dir = Path(args.input)
        if processed_dir.exists():
            features_df = pl.read_parquet(processed_dir / "sensor_features.parquet")
            survey_df = pl.read_parquet(processed_dir / "survey_scores.parquet")
            G = add_user_nodes(G, features_df, survey_df)
            logger.info(f"After adding users: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Graph statistics
    logger.info("\n" + "=" * 60)
    logger.info("GRAPH STATISTICS")
    logger.info("=" * 60)

    node_types = {}
    for _, attrs in G.nodes(data=True):
        nt = attrs.get("node_type", "unknown")
        node_types[nt] = node_types.get(nt, 0) + 1

    for nt, count in sorted(node_types.items()):
        logger.info(f"  {nt}: {count} nodes")

    edge_types = {}
    for _, _, attrs in G.edges(data=True):
        et = attrs.get("edge_type", "unknown")
        edge_types[et] = edge_types.get(et, 0) + 1

    for et, count in sorted(edge_types.items()):
        logger.info(f"  {et}: {count} edges")

    # Export graph
    export_graph(G, output_dir / "knowledge_graph.json")

    # Visualize
    visualize_knowledge_graph(G, output_dir / "kg_visualization.png")

    logger.info("\nKnowledge graph construction complete!")


if __name__ == "__main__":
    main()
