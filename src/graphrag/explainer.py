"""
Clinical Report Generator using GraphRAG.

Generates interpretable mental health assessment reports by:
1. Retrieving relevant knowledge subgraph
2. Formatting evidence and predictions
3. Using LLM to generate natural language explanation

IMPORTANT: Generated reports are for RESEARCH ONLY.
They are NOT clinical diagnoses.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from string import Template

from loguru import logger

from .llm_client import LLMClient


@dataclass
class UserProfile:
    """User data for report generation."""

    user_id: str
    sensor_features: Dict[str, float]
    survey_scores: Dict[str, float]
    predictions: Dict[str, float]
    attention_weights: Optional[Dict[str, float]] = None


@dataclass
class KnowledgeContext:
    """Retrieved knowledge for report generation."""

    relevant_symptoms: List[str]
    behavior_symptom_paths: List[Dict]
    statistical_evidence: List[Dict]
    literature_references: List[str]


class ClinicalReportGenerator:
    """
    Generates clinical-style reports from GNN predictions.

    The report includes:
    1. Risk assessment summary
    2. Key digital biomarker findings
    3. Evidence chain from behavior to symptom
    4. Statistical support
    5. Limitations and disclaimers
    """

    SYSTEM_PROMPT = """You are a research assistant helping to interpret mental health
prediction results from a machine learning model. Your task is to generate clear,
accurate, and scientifically grounded explanations.

IMPORTANT GUIDELINES:
1. This is RESEARCH ONLY - never claim diagnostic capability
2. Always cite statistical evidence (correlation, p-value, effect size)
3. Distinguish between correlation and causation
4. Acknowledge limitations (sample size, generalizability)
5. Use appropriate hedging language ("may indicate", "suggests", "associated with")
6. Be culturally sensitive and avoid stigmatizing language

Format your response in clear Markdown with appropriate headers."""

    REPORT_TEMPLATE = Template("""
## Mental Health Assessment Report - User ${user_id}

### 1. Risk Assessment Summary

**Predicted PHQ-9 Score**: ${phq9_score} (${severity_level})
**Model Confidence**: ${confidence}%

${risk_interpretation}

### 2. Key Digital Biomarker Findings

${biomarker_findings}

### 3. Evidence Chain

${evidence_chain}

### 4. Statistical Support

${statistical_support}

### 5. Modality Contributions

${modality_contributions}

### 6. Limitations

${limitations}

### References

${references}

---
*This report is generated for research purposes only and does not constitute
a clinical diagnosis. Please consult a mental health professional for clinical evaluation.*
""")

    def __init__(self, llm_client: LLMClient):
        """
        Initialize report generator.

        Args:
            llm_client: LLM client for text generation
        """
        self.llm = llm_client

    def generate_report(
        self,
        user_profile: UserProfile,
        knowledge_context: KnowledgeContext,
        use_template: bool = True,
    ) -> str:
        """
        Generate a clinical-style report for a user.

        Args:
            user_profile: User data and predictions
            knowledge_context: Retrieved knowledge
            use_template: If True, use structured template; else free-form LLM

        Returns:
            Formatted Markdown report
        """
        if use_template:
            return self._generate_with_template(user_profile, knowledge_context)
        else:
            return self._generate_with_llm(user_profile, knowledge_context)

    def _generate_with_template(
        self,
        user: UserProfile,
        knowledge: KnowledgeContext,
    ) -> str:
        """Generate report using structured template."""

        # Severity interpretation
        phq9_score = user.predictions.get("phq9_score", 0)
        severity = self._get_severity_level(phq9_score)
        risk_interp = self._interpret_risk(phq9_score, severity)

        # Biomarker findings
        biomarkers = self._format_biomarker_findings(
            user.sensor_features, knowledge.behavior_symptom_paths
        )

        # Evidence chain
        evidence = self._format_evidence_chain(knowledge.behavior_symptom_paths)

        # Statistical support
        stats = self._format_statistical_support(knowledge.statistical_evidence)

        # Modality contributions
        modality = self._format_modality_contributions(user.attention_weights)

        # Limitations
        limitations = self._format_limitations()

        # References
        refs = self._format_references(knowledge.literature_references)

        # Fill template
        report = self.REPORT_TEMPLATE.substitute(
            user_id=user.user_id,
            phq9_score=f"{phq9_score:.1f}",
            severity_level=severity,
            confidence=f"{user.predictions.get('confidence', 75):.0f}",
            risk_interpretation=risk_interp,
            biomarker_findings=biomarkers,
            evidence_chain=evidence,
            statistical_support=stats,
            modality_contributions=modality,
            limitations=limitations,
            references=refs,
        )

        return report

    def _generate_with_llm(
        self,
        user: UserProfile,
        knowledge: KnowledgeContext,
    ) -> str:
        """Generate report using LLM for more natural language."""

        # Prepare context for LLM
        prompt = f"""Generate a clinical-style research report for user {user.user_id}.

## User Data

### Sensor Features (normalized, higher = more of behavior)
{self._format_features_for_llm(user.sensor_features)}

### Survey Scores
{self._format_scores_for_llm(user.survey_scores)}

### Model Predictions
- PHQ-9 Score: {user.predictions.get('phq9_score', 'N/A'):.1f}
- Predicted Severity: {self._get_severity_level(user.predictions.get('phq9_score', 0))}

## Retrieved Knowledge

### Relevant Symptoms
{', '.join(knowledge.relevant_symptoms)}

### Behavior-Symptom Mappings
{self._format_paths_for_llm(knowledge.behavior_symptom_paths)}

### Statistical Evidence
{self._format_stats_for_llm(knowledge.statistical_evidence)}

## Instructions

Generate a comprehensive but concise report that:
1. Summarizes the risk level
2. Explains key behavioral findings
3. Links behaviors to symptoms using the knowledge graph
4. Cites statistical evidence
5. Acknowledges limitations

Use clear headers and bullet points. Be scientifically accurate but accessible."""

        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.3,  # Lower temperature for factual report
            max_tokens=2000,
        )

        return response

    def _get_severity_level(self, phq9_score: float) -> str:
        """Convert PHQ-9 score to severity category."""
        if phq9_score <= 4:
            return "Minimal"
        elif phq9_score <= 9:
            return "Mild"
        elif phq9_score <= 14:
            return "Moderate"
        elif phq9_score <= 19:
            return "Moderately Severe"
        else:
            return "Severe"

    def _interpret_risk(self, score: float, severity: str) -> str:
        """Generate risk interpretation text."""
        if severity == "Minimal":
            return (
                "The predicted PHQ-9 score falls within the minimal range, "
                "suggesting low depression symptom burden. However, subclinical "
                "symptoms may still warrant monitoring."
            )
        elif severity == "Mild":
            return (
                "The predicted score indicates mild depression symptoms. "
                "This may reflect normal stress responses or early symptoms. "
                "Watchful waiting and lifestyle interventions may be appropriate."
            )
        elif severity == "Moderate":
            return (
                "The predicted score indicates moderate depression symptoms. "
                "This level often warrants clinical evaluation and may benefit "
                "from professional intervention."
            )
        else:
            return (
                "The predicted score indicates elevated depression symptoms. "
                "Professional clinical evaluation is strongly recommended. "
                "Note: This is a research prediction, not a diagnosis."
            )

    def _format_biomarker_findings(
        self,
        features: Dict[str, float],
        paths: List[Dict],
    ) -> str:
        """Format biomarker findings section."""
        lines = []

        # Sort features by absolute deviation from mean (assuming normalized)
        sorted_features = sorted(
            features.items(), key=lambda x: abs(x[1] - 0.5), reverse=True
        )

        for feature, value in sorted_features[:5]:  # Top 5 notable features
            # Find related symptoms
            related_symptoms = []
            for path in paths:
                if path.get("behavior") == feature:
                    related_symptoms.append(path.get("symptom", ""))

            symptom_text = (
                f" (associated with: {', '.join(related_symptoms)})"
                if related_symptoms
                else ""
            )

            # Interpret value
            if value > 0.7:
                level = "elevated"
            elif value < 0.3:
                level = "reduced"
            else:
                level = "within normal range"

            lines.append(f"- **{feature}**: {level} ({value:.2f}){symptom_text}")

        return "\n".join(lines) if lines else "No notable biomarker deviations detected."

    def _format_evidence_chain(self, paths: List[Dict]) -> str:
        """Format behavior -> symptom evidence chains."""
        if not paths:
            return "No behavior-symptom mappings retrieved."

        lines = ["```"]
        for path in paths[:5]:  # Top 5 paths
            behavior = path.get("behavior", "?")
            symptom = path.get("symptom", "?")
            evidence = path.get("evidence", "weak")

            lines.append(f"{behavior} --[{evidence}]--> {symptom}")

        lines.append("```")
        return "\n".join(lines)

    def _format_statistical_support(self, evidence: List[Dict]) -> str:
        """Format statistical evidence."""
        if not evidence:
            return "Statistical evidence not available."

        lines = ["| Relationship | r | p-adj | Effect |", "|---|---|---|---|"]

        for ev in evidence[:10]:  # Top 10
            relationship = f"{ev.get('feature', '?')} → {ev.get('outcome', '?')}"
            r = ev.get("correlation", 0)
            p = ev.get("p_adjusted", 1)
            effect = ev.get("interpretation", "?")

            lines.append(f"| {relationship} | {r:.2f} | {p:.3f} | {effect} |")

        return "\n".join(lines)

    def _format_modality_contributions(
        self, attention: Optional[Dict[str, float]]
    ) -> str:
        """Format modality attention weights."""
        if not attention:
            return "Modality contribution data not available."

        lines = []
        for modality, weight in sorted(
            attention.items(), key=lambda x: x[1], reverse=True
        ):
            bar = "█" * int(weight * 20)
            lines.append(f"- **{modality}**: {bar} ({weight:.1%})")

        return "\n".join(lines)

    def _format_limitations(self) -> str:
        """Standard limitations text."""
        return """
- **Sample Size**: Model trained on ~50 participants; generalizability is limited
- **Population**: College students only; may not apply to other demographics
- **Temporal**: Cross-sectional prediction; not validated longitudinally
- **Causality**: Correlational relationships only; not causal claims
- **Clinical Validity**: Research tool only; requires clinical validation
"""

    def _format_references(self, refs: List[str]) -> str:
        """Format literature references."""
        if not refs:
            return "- Wang et al. (2014). StudentLife Study\n- DSM-5 (2013). Diagnostic Criteria"

        lines = [f"- {ref}" for ref in refs]
        return "\n".join(lines)

    def _format_features_for_llm(self, features: Dict[str, float]) -> str:
        """Format features for LLM prompt."""
        lines = []
        for name, value in sorted(features.items()):
            lines.append(f"- {name}: {value:.2f}")
        return "\n".join(lines)

    def _format_scores_for_llm(self, scores: Dict[str, float]) -> str:
        """Format survey scores for LLM prompt."""
        lines = []
        for name, value in sorted(scores.items()):
            lines.append(f"- {name}: {value:.1f}")
        return "\n".join(lines)

    def _format_paths_for_llm(self, paths: List[Dict]) -> str:
        """Format knowledge paths for LLM prompt."""
        lines = []
        for path in paths:
            lines.append(
                f"- {path.get('behavior', '?')} → {path.get('symptom', '?')} "
                f"(evidence: {path.get('evidence', '?')})"
            )
        return "\n".join(lines)

    def _format_stats_for_llm(self, stats: List[Dict]) -> str:
        """Format statistical evidence for LLM prompt."""
        lines = []
        for stat in stats:
            lines.append(
                f"- {stat.get('feature', '?')} correlates with {stat.get('outcome', '?')} "
                f"(r={stat.get('correlation', 0):.2f}, p={stat.get('p_adjusted', 1):.3f})"
            )
        return "\n".join(lines)


class InteractiveQA:
    """
    Interactive Q&A system for exploring predictions.

    Allows users to ask questions about:
    - Why a specific user was predicted high/low risk
    - What behaviors are most concerning
    - How predictions would change with different data
    """

    SYSTEM_PROMPT = """You are a research assistant helping users understand
mental health prediction results. Answer questions accurately based on the
provided data and knowledge graph. Always:

1. Ground answers in the provided evidence
2. Acknowledge uncertainty when appropriate
3. Never make clinical diagnoses
4. Suggest consulting professionals for clinical questions
5. Be empathetic but scientifically accurate"""

    def __init__(
        self,
        llm_client: LLMClient,
        knowledge_graph: Any,  # NetworkX graph or similar
    ):
        """
        Initialize QA system.

        Args:
            llm_client: LLM for generating answers
            knowledge_graph: Knowledge graph for retrieval
        """
        self.llm = llm_client
        self.kg = knowledge_graph
        self.conversation_history: List[Dict[str, str]] = []

    def ask(
        self,
        question: str,
        user_context: Optional[UserProfile] = None,
    ) -> str:
        """
        Answer a question about the predictions.

        Args:
            question: User's question
            user_context: Optional user data for context

        Returns:
            Answer string
        """
        # Build context
        context = self._build_context(question, user_context)

        # Add to conversation
        self.conversation_history.append({"role": "user", "content": question})

        # Prepare messages
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Add context as system message
        if context:
            messages.append(
                {"role": "system", "content": f"Relevant context:\n{context}"}
            )

        # Add conversation history (last 5 turns)
        messages.extend(self.conversation_history[-10:])

        # Generate response
        response = self.llm.chat(messages, temperature=0.5, max_tokens=1000)

        # Add to history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def _build_context(
        self,
        question: str,
        user_context: Optional[UserProfile],
    ) -> str:
        """Build relevant context for the question."""
        context_parts = []

        # Add user data if available
        if user_context:
            context_parts.append(f"User ID: {user_context.user_id}")
            context_parts.append(
                f"Predicted PHQ-9: {user_context.predictions.get('phq9_score', 'N/A')}"
            )

            # Add notable features
            if user_context.sensor_features:
                notable = sorted(
                    user_context.sensor_features.items(),
                    key=lambda x: abs(x[1] - 0.5),
                    reverse=True,
                )[:3]
                context_parts.append(
                    f"Notable features: {', '.join(f'{k}={v:.2f}' for k, v in notable)}"
                )

        # Retrieve from knowledge graph
        # (Simplified - full implementation would use embedding similarity)
        if self.kg is not None:
            try:
                import networkx as nx

                # Get relevant nodes based on keywords in question
                keywords = question.lower().split()
                relevant_nodes = []

                for node in self.kg.nodes():
                    node_str = str(node).lower()
                    if any(kw in node_str for kw in keywords):
                        relevant_nodes.append(node)

                if relevant_nodes:
                    context_parts.append(f"Related concepts: {', '.join(relevant_nodes[:5])}")

            except Exception as e:
                logger.warning(f"KG retrieval failed: {e}")

        return "\n".join(context_parts)

    def reset(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []
