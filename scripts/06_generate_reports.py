#!/usr/bin/env python3
"""
Step 6: Report Generation

Generates clinical-style reports using GraphRAG + LLM.

Supports: CUDA, MPS (Apple Silicon), CPU (auto-detected)

Usage:
    python scripts/06_generate_reports.py --user_id u23
    python scripts/06_generate_reports.py --all

Outputs:
    - outputs/reports/report_{user_id}.md
    - outputs/reports/report_{user_id}.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import polars as pl
from loguru import logger

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gnn.encoders import MentalHealthGNN
from src.graphrag.explainer import ClinicalReportGenerator, UserProfile, KnowledgeContext
from src.graphrag.llm_client import get_llm_client
from src.knowledge.schemas.dsm5 import (
    MDDSymptom,
    get_symptoms_for_behavior,
    DSM5_BEHAVIOR_MAPPINGS,
    DIGITAL_BEHAVIORS
)


def get_device(requested: str = "auto") -> torch.device:
    """Auto-detect best available device."""
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    elif requested == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            logger.warning("MPS requested but not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def setup_logging(output_dir: Path) -> None:
    """Configure logging."""
    log_file = output_dir / "logs" / f"reports_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_file, rotation="10 MB", level="DEBUG")


def load_model(model_path: Path, device: torch.device) -> tuple:
    """Load trained GNN model."""
    checkpoint = torch.load(model_path, map_location=device)

    model = MentalHealthGNN(
        node_features=checkpoint["n_features"],
        **checkpoint["model_config"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint["feature_cols"]


def load_knowledge_graph(kg_path: Path) -> dict:
    """Load knowledge graph."""
    with open(kg_path) as f:
        return json.load(f)


def load_user_data(
    processed_dir: Path,
    user_id: str
) -> Optional[Dict]:
    """Load data for a specific user."""
    features_df = pl.read_parquet(processed_dir / "sensor_features.parquet")
    survey_df = pl.read_parquet(processed_dir / "survey_scores.parquet")

    # Get user features
    user_features = features_df.filter(pl.col("user_id") == user_id)
    if len(user_features) == 0:
        return None

    # Get user survey scores
    pre_df = survey_df.filter(pl.col("type") == "pre")
    user_survey = pre_df.filter(pl.col("uid") == user_id)

    data = {
        "user_id": user_id,
        "features": {},
        "survey_scores": {}
    }

    # Extract features
    exclude_cols = ["user_id", "gps_valid", "activity_valid", "phone_valid", "conv_valid"]
    for col in features_df.columns:
        if col not in exclude_cols:
            val = user_features[col].to_list()[0]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                data["features"][col] = float(val)

    # Extract survey scores
    if len(user_survey) > 0:
        for col in user_survey.columns:
            if col not in ["uid", "type"]:
                val = user_survey[col].to_list()[0]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    data["survey_scores"][col] = float(val)

    return data


def predict_for_user(
    model: MentalHealthGNN,
    user_data: Dict,
    feature_cols: List[str],
    device: torch.device
) -> Dict:
    """Get model predictions for a user."""
    # Create feature vector
    feature_vector = []
    for col in feature_cols:
        feature_vector.append(user_data["features"].get(col, 0.0))

    X = torch.tensor([feature_vector], dtype=torch.float32).to(device)

    # Create minimal graph (single node)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)

    with torch.no_grad():
        data = {
            "sensor_features": X,
            "edge_index": edge_index
        }
        outputs = model(data)

    predictions = {
        "phq9_score": float(outputs["phq9_score"].cpu().numpy()[0, 0]),
        "confidence": 75.0  # Placeholder - could compute from model uncertainty
    }

    return predictions


def retrieve_knowledge_context(
    kg_data: dict,
    user_data: Dict,
    statistical_results: Optional[dict] = None
) -> KnowledgeContext:
    """Retrieve relevant knowledge for a user."""
    # Find relevant symptoms based on user's notable behaviors
    relevant_symptoms = set()
    behavior_paths = []

    # Identify notable features (those deviating from typical values)
    for feature, value in user_data["features"].items():
        symptoms = get_symptoms_for_behavior(feature)
        for symptom in symptoms:
            relevant_symptoms.add(symptom.value)

            # Create behavior-symptom path
            behavior_paths.append({
                "behavior": feature,
                "symptom": symptom.value,
                "evidence": "moderate",  # Could look up from DSM5_BEHAVIOR_MAPPINGS
                "value": value
            })

    # Get statistical evidence from relations file
    statistical_evidence = []
    if statistical_results:
        for rel in statistical_results.get("relations", []):
            if rel["sensor_feature"] in user_data["features"]:
                statistical_evidence.append({
                    "feature": rel["sensor_feature"],
                    "outcome": rel["survey_item"],
                    "correlation": rel.get("correlation", 0),
                    "p_adjusted": rel.get("p_adjusted", 1),
                    "interpretation": rel.get("interpretation", "")
                })

    # Literature references
    references = [
        "Wang et al. (2014). StudentLife: Assessing Mental Health and Academic Performance of College Students Using Smartphones",
        "Saeb et al. (2015). Mobile Phone Sensor Correlates of Depressive Symptom Severity",
        "DSM-5 (2013). Diagnostic and Statistical Manual of Mental Disorders, Fifth Edition"
    ]

    return KnowledgeContext(
        relevant_symptoms=list(relevant_symptoms),
        behavior_symptom_paths=behavior_paths[:10],  # Top 10
        statistical_evidence=statistical_evidence[:10],
        literature_references=references
    )


def generate_report_for_user(
    user_id: str,
    model: MentalHealthGNN,
    feature_cols: List[str],
    kg_data: dict,
    processed_dir: Path,
    statistical_results: Optional[dict],
    output_dir: Path,
    device: torch.device,
    llm_client=None
) -> Optional[str]:
    """Generate report for a single user."""
    # Load user data
    user_data = load_user_data(processed_dir, user_id)
    if user_data is None:
        logger.warning(f"User {user_id} not found")
        return None

    # Get predictions
    predictions = predict_for_user(model, user_data, feature_cols, device)

    # Create user profile
    profile = UserProfile(
        user_id=user_id,
        sensor_features=user_data["features"],
        survey_scores=user_data["survey_scores"],
        predictions=predictions,
        attention_weights=None  # Could extract from model
    )

    # Retrieve knowledge context
    knowledge = retrieve_knowledge_context(kg_data, user_data, statistical_results)

    # Generate report
    if llm_client:
        generator = ClinicalReportGenerator(llm_client)
        report = generator.generate_report(profile, knowledge, use_template=False)
    else:
        # Use template-based generation without LLM
        generator = ClinicalReportGenerator(None)
        report = generator._generate_with_template(profile, knowledge)

    # Save report
    report_md_path = output_dir / f"report_{user_id}.md"
    with open(report_md_path, "w") as f:
        f.write(report)

    # Save structured data
    report_json_path = output_dir / f"report_{user_id}.json"
    report_data = {
        "user_id": user_id,
        "generated": datetime.now().isoformat(),
        "predictions": predictions,
        "features": user_data["features"],
        "survey_scores": user_data["survey_scores"],
        "relevant_symptoms": knowledge.relevant_symptoms,
        "behavior_paths": knowledge.behavior_symptom_paths
    }
    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"Generated report for {user_id}: {report_md_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate clinical reports")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/processed",
        help="Input directory with preprocessed data",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/models/gnn_best.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--kg",
        type=str,
        default="outputs/graphs/knowledge_graph.json",
        help="Path to knowledge graph",
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
        default="outputs/reports",
        help="Output directory",
    )
    parser.add_argument(
        "--user_id",
        type=str,
        help="Generate report for specific user",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate reports for all users",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for report generation (requires API key)",
    )
    parser.add_argument(
        "--llm-type",
        type=str,
        default="deepseek",
        choices=["deepseek", "local_qwen"],
        help="LLM type to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    device = get_device(args.device)

    logger.info("=" * 60)
    logger.info("Report Generation")
    logger.info("=" * 60)

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Run 05_train_gnn.py first")
        return

    model, feature_cols = load_model(model_path, device)
    logger.info(f"Loaded model with {len(feature_cols)} features")

    # Load knowledge graph
    kg_path = Path(args.kg)
    if kg_path.exists():
        kg_data = load_knowledge_graph(kg_path)
        logger.info(f"Loaded KG with {len(kg_data['nodes'])} nodes")
    else:
        logger.warning(f"Knowledge graph not found: {kg_path}")
        kg_data = {"nodes": [], "edges": []}

    # Load statistical results
    relations_path = Path(args.relations)
    statistical_results = None
    if relations_path.exists():
        with open(relations_path) as f:
            statistical_results = json.load(f)

    # Initialize LLM client if requested
    llm_client = None
    if args.use_llm:
        import os
        if args.llm_type == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                logger.warning("DEEPSEEK_API_KEY not set, using template-based generation")
            else:
                llm_client = get_llm_client({"type": "deepseek", "api_key": api_key})
        else:
            llm_client = get_llm_client({"type": "local_qwen"})

    # Get users to process
    processed_dir = Path(args.input)
    features_df = pl.read_parquet(processed_dir / "sensor_features.parquet")
    all_user_ids = features_df["user_id"].to_list()

    if args.user_id:
        user_ids = [args.user_id]
    elif args.all:
        user_ids = all_user_ids
    else:
        # Default: first 5 users
        user_ids = all_user_ids[:5]
        logger.info(f"Generating reports for first 5 users. Use --all for all users.")

    # Generate reports
    for user_id in user_ids:
        generate_report_for_user(
            user_id,
            model,
            feature_cols,
            kg_data,
            processed_dir,
            statistical_results,
            output_dir,
            device,
            llm_client
        )

    logger.info(f"\nGenerated {len(user_ids)} reports in {output_dir}")


if __name__ == "__main__":
    main()
