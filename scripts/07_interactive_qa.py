#!/usr/bin/env python3
"""
Step 7: Interactive Q&A System

Interactive question-answering system for exploring predictions.

Supports: CUDA, MPS (Apple Silicon), CPU (auto-detected)

Usage:
    python scripts/07_interactive_qa.py
    python scripts/07_interactive_qa.py --user_id u23

Example questions:
    > Why is u23 predicted as moderate depression risk?
    > What behaviors are most concerning for u23?
    > Which users have similar patterns to u23?
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import polars as pl
from loguru import logger

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gnn.encoders import MentalHealthGNN
from src.graphrag.explainer import InteractiveQA, UserProfile
from src.graphrag.llm_client import get_llm_client
from src.knowledge.schemas.dsm5 import get_symptoms_for_behavior


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
    log_file = output_dir / "logs" / f"qa_{datetime.now():%Y%m%d_%H%M%S}.log"
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


def load_knowledge_graph_as_nx(kg_path: Path):
    """Load knowledge graph as NetworkX graph."""
    import networkx as nx

    with open(kg_path) as f:
        kg_data = json.load(f)

    G = nx.DiGraph()

    for node in kg_data["nodes"]:
        node_id = node.pop("id")
        G.add_node(node_id, **node)

    for edge in kg_data["edges"]:
        source = edge.pop("source")
        target = edge.pop("target")
        G.add_edge(source, target, **edge)

    return G


def load_user_data(processed_dir: Path, user_id: str) -> Optional[Dict]:
    """Load data for a specific user."""
    features_df = pl.read_parquet(processed_dir / "sensor_features.parquet")
    survey_df = pl.read_parquet(processed_dir / "survey_scores.parquet")

    user_features = features_df.filter(pl.col("user_id") == user_id)
    if len(user_features) == 0:
        return None

    pre_df = survey_df.filter(pl.col("type") == "pre")
    user_survey = pre_df.filter(pl.col("uid") == user_id)

    data = {"user_id": user_id, "features": {}, "survey_scores": {}}

    exclude_cols = ["user_id", "gps_valid", "activity_valid", "phone_valid", "conv_valid"]
    for col in features_df.columns:
        if col not in exclude_cols:
            val = user_features[col].to_list()[0]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                data["features"][col] = float(val)

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
    feature_cols: list,
    device: torch.device
) -> Dict:
    """Get model predictions for a user."""
    feature_vector = []
    for col in feature_cols:
        feature_vector.append(user_data["features"].get(col, 0.0))

    X = torch.tensor([feature_vector], dtype=torch.float32).to(device)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)

    with torch.no_grad():
        data = {"sensor_features": X, "edge_index": edge_index}
        outputs = model(data)

    return {
        "phq9_score": float(outputs["phq9_score"].cpu().numpy()[0, 0]),
        "confidence": 75.0
    }


class SimpleQASystem:
    """
    Simple Q&A system without LLM.

    Provides rule-based answers for common questions.
    """

    def __init__(
        self,
        model: MentalHealthGNN,
        feature_cols: list,
        kg_data: dict,
        processed_dir: Path,
        device: torch.device
    ):
        self.model = model
        self.feature_cols = feature_cols
        self.kg_data = kg_data
        self.processed_dir = processed_dir
        self.device = device
        self.current_user = None
        self.current_user_data = None

    def set_user(self, user_id: str) -> bool:
        """Set the current user context."""
        user_data = load_user_data(self.processed_dir, user_id)
        if user_data is None:
            return False
        self.current_user = user_id
        self.current_user_data = user_data
        return True

    def answer(self, question: str) -> str:
        """Answer a question."""
        question_lower = question.lower()

        # Check for user-specific questions
        if self.current_user and self.current_user_data:
            if "why" in question_lower and ("risk" in question_lower or "predict" in question_lower):
                return self._explain_prediction()
            elif "behavior" in question_lower or "concerning" in question_lower:
                return self._explain_behaviors()
            elif "similar" in question_lower:
                return self._find_similar_users()
            elif "symptom" in question_lower:
                return self._explain_symptoms()

        # General questions
        if "help" in question_lower or "?" in question_lower and len(question_lower) < 10:
            return self._help_message()

        return "I'm not sure how to answer that. Try asking about predictions, behaviors, or symptoms."

    def _explain_prediction(self) -> str:
        """Explain why a user was predicted at certain risk level."""
        predictions = predict_for_user(
            self.model, self.current_user_data, self.feature_cols, self.device
        )
        phq9 = predictions["phq9_score"]

        if phq9 <= 4:
            severity = "minimal"
        elif phq9 <= 9:
            severity = "mild"
        elif phq9 <= 14:
            severity = "moderate"
        elif phq9 <= 19:
            severity = "moderately severe"
        else:
            severity = "severe"

        lines = [
            f"## Prediction Explanation for {self.current_user}",
            "",
            f"**Predicted PHQ-9 Score:** {phq9:.1f} ({severity})",
            "",
            "**Key Contributing Factors:**"
        ]

        # Identify notable features
        features = self.current_user_data["features"]
        sorted_features = sorted(
            features.items(),
            key=lambda x: abs(x[1] - 0.5) if x[1] <= 1 else abs(x[1]),
            reverse=True
        )

        for feature, value in sorted_features[:5]:
            symptoms = get_symptoms_for_behavior(feature)
            symptom_str = ", ".join(s.value for s in symptoms) if symptoms else "general"
            lines.append(f"- **{feature}**: {value:.2f} (related to: {symptom_str})")

        lines.extend([
            "",
            "**Note:** This is a research prediction, not a clinical diagnosis."
        ])

        return "\n".join(lines)

    def _explain_behaviors(self) -> str:
        """Explain concerning behaviors."""
        lines = [
            f"## Behavioral Analysis for {self.current_user}",
            ""
        ]

        features = self.current_user_data["features"]

        # Identify concerning behaviors
        concerning = []
        for feature, value in features.items():
            symptoms = get_symptoms_for_behavior(feature)
            if symptoms:
                concerning.append({
                    "feature": feature,
                    "value": value,
                    "symptoms": [s.value for s in symptoms]
                })

        if concerning:
            lines.append("**Behaviors linked to depression symptoms:**")
            for item in concerning[:7]:
                lines.append(
                    f"- **{item['feature']}** ({item['value']:.2f}): "
                    f"associated with {', '.join(item['symptoms'])}"
                )
        else:
            lines.append("No strongly concerning behaviors identified.")

        return "\n".join(lines)

    def _explain_symptoms(self) -> str:
        """Explain relevant symptoms."""
        lines = [
            f"## Symptom Mapping for {self.current_user}",
            ""
        ]

        features = self.current_user_data["features"]
        symptom_behaviors = {}

        for feature in features:
            symptoms = get_symptoms_for_behavior(feature)
            for symptom in symptoms:
                if symptom.value not in symptom_behaviors:
                    symptom_behaviors[symptom.value] = []
                symptom_behaviors[symptom.value].append(feature)

        if symptom_behaviors:
            lines.append("**DSM-5 symptoms and their digital behavior proxies:**")
            for symptom, behaviors in symptom_behaviors.items():
                lines.append(f"- **{symptom}**: {', '.join(behaviors)}")
        else:
            lines.append("No symptom mappings found for this user's behaviors.")

        return "\n".join(lines)

    def _find_similar_users(self) -> str:
        """Find users with similar patterns."""
        # Load all user data
        features_df = pl.read_parquet(self.processed_dir / "sensor_features.parquet")

        exclude_cols = ["user_id", "gps_valid", "activity_valid", "phone_valid", "conv_valid"]
        feature_cols = [c for c in features_df.columns if c not in exclude_cols]

        # Compute similarity
        current_vector = np.array([
            self.current_user_data["features"].get(c, 0) for c in feature_cols
        ])

        similarities = []
        for row in features_df.iter_rows(named=True):
            if row["user_id"] == self.current_user:
                continue
            other_vector = np.array([row.get(c, 0) or 0 for c in feature_cols])

            # Cosine similarity
            norm_current = np.linalg.norm(current_vector)
            norm_other = np.linalg.norm(other_vector)
            if norm_current > 0 and norm_other > 0:
                sim = np.dot(current_vector, other_vector) / (norm_current * norm_other)
                similarities.append((row["user_id"], sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        lines = [
            f"## Users Similar to {self.current_user}",
            "",
            "**Top 5 most similar users:**"
        ]

        for user_id, sim in similarities[:5]:
            lines.append(f"- **{user_id}**: similarity = {sim:.3f}")

        return "\n".join(lines)

    def _help_message(self) -> str:
        """Return help message."""
        return """
## Interactive Q&A Help

**Available commands:**
- `user <user_id>` - Set current user context
- `quit` or `exit` - Exit the system

**Example questions:**
- "Why is this user predicted as high risk?"
- "What behaviors are concerning?"
- "Which symptoms are relevant?"
- "Who are similar users?"

**Current user:** """ + (self.current_user or "None (use 'user <id>' to set)")


def run_interactive_session(
    qa_system,
    initial_user: Optional[str] = None
):
    """Run interactive Q&A session."""
    print("\n" + "=" * 60)
    print("Mental Health Prediction Q&A System")
    print("=" * 60)
    print("\nType 'help' for available commands, 'quit' to exit.")

    if initial_user:
        if qa_system.set_user(initial_user):
            print(f"\nUser context set to: {initial_user}")
        else:
            print(f"\nWarning: User {initial_user} not found")

    while True:
        try:
            question = input("\n> ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            # Handle user command
            if question.lower().startswith("user "):
                user_id = question[5:].strip()
                if qa_system.set_user(user_id):
                    print(f"User context set to: {user_id}")
                else:
                    print(f"User {user_id} not found")
                continue

            # Get answer
            answer = qa_system.answer(question)
            print("\n" + answer)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Interactive Q&A system")
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
        "--user_id",
        type=str,
        help="Initial user to analyze",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for answers (requires API key)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Setup
    np.random.seed(42)
    torch.manual_seed(42)
    output_dir = Path("outputs/qa")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    device = get_device(args.device)

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        print(f"Error: Model not found at {model_path}")
        print("Run 05_train_gnn.py first")
        return

    model, feature_cols = load_model(model_path, device)

    # Load knowledge graph
    kg_path = Path(args.kg)
    kg_data = {"nodes": [], "edges": []}
    if kg_path.exists():
        with open(kg_path) as f:
            kg_data = json.load(f)

    # Create Q&A system
    if args.use_llm:
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            kg_nx = load_knowledge_graph_as_nx(kg_path) if kg_path.exists() else None
            llm_client = get_llm_client({"type": "deepseek", "api_key": api_key})

            # Load user profile if specified
            user_profile = None
            if args.user_id:
                user_data = load_user_data(Path(args.input), args.user_id)
                if user_data:
                    predictions = predict_for_user(model, user_data, feature_cols, device)
                    user_profile = UserProfile(
                        user_id=args.user_id,
                        sensor_features=user_data["features"],
                        survey_scores=user_data["survey_scores"],
                        predictions=predictions
                    )

            qa_system = InteractiveQA(llm_client, kg_nx)
            # Run LLM-based interactive session
            print("\n" + "=" * 60)
            print("Mental Health Prediction Q&A System (LLM-powered)")
            print("=" * 60)

            while True:
                try:
                    question = input("\n> ").strip()
                    if question.lower() in ["quit", "exit", "q"]:
                        break
                    answer = qa_system.ask(question, user_profile)
                    print("\n" + answer)
                except (KeyboardInterrupt, EOFError):
                    break
        else:
            print("DEEPSEEK_API_KEY not set, using simple Q&A system")
            qa_system = SimpleQASystem(
                model, feature_cols, kg_data, Path(args.input), device
            )
            run_interactive_session(qa_system, args.user_id)
    else:
        qa_system = SimpleQASystem(
            model, feature_cols, kg_data, Path(args.input), device
        )
        run_interactive_session(qa_system, args.user_id)


if __name__ == "__main__":
    main()
