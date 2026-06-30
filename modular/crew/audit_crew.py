"""CrewAI orchestration for the GASCO ML audit workflow."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import warnings

import pandas as pd

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM

from modular.agents.coverage_agent import CoverageAgent, CoverageResult
from modular.agents.instruction_agent import InstructionAgent, InstructionResult
from modular.agents.risk_discovery_agent import (
    RISK_COLUMNS,
    build_official_risk_discovery_input,
    discover_risks,
)
from modular.agents.risk_prediction_agent import RiskPredictionAgent, RiskPredictionResult
from modular.bdo.component_instruction_pack import ComponentInstructionPack
from modular.bdo.documentation_memo import BDODocumentationMemo
from modular.bdo.guardrails import BDOGuardrails, GuardrailCoverageStatus
from modular.bdo.methodology_mapper import BDOMethodologyMapper
from modular.config.config_schema import AppConfig
from modular.data.loader import DataLoader
from modular.export.csv_exporter import CSVExporter
from modular.hitl.review_layer import HumanInTheLoopReviewLayer, HumanReviewArtifacts
from modular.model.explainability import ScopeModelExplainability
from modular.tools.ml_scope_prediction_tool import MLScopePredictionTool

warnings.filterwarnings(
    "ignore",
    message="method callbacks cannot be serialized.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="function callbacks cannot be serialized.*",
    category=UserWarning,
)


class LocalCrewLLM(BaseLLM):
    """Deterministic local LLM shim so CrewAI runs without external services."""

    def __init__(self):
        super().__init__(model="gasco-local-deterministic", provider="local")

    def call(
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[Any] | None = None,
    ) -> str:
        task_name = getattr(from_task, "name", "CrewAI task")
        agent_role = getattr(from_agent, "role", "GASCO agent")
        return f"{task_name} completed by {agent_role}. Local deterministic stage callback produced the audit artifact."


class GASCOAuditCrew:
    """CrewAI workflow that delegates predictions to MLScopeEngine."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.output_directory = Path(config.output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.data_loader = DataLoader()
        self.csv_exporter = CSVExporter(self.output_directory)
        self.prediction_tool = MLScopePredictionTool(
            findings_data_path=config.findings_data_path
        )
        self.risk_prediction_agent = RiskPredictionAgent(self.prediction_tool)
        self.bdo_guardrails = BDOGuardrails(config.bdo_guardrails)
        self.bdo_methodology_mapper = BDOMethodologyMapper()
        self.bdo_documentation_memo = BDODocumentationMemo()
        self.component_instruction_pack = ComponentInstructionPack()
        self.human_review_layer = HumanInTheLoopReviewLayer()
        self.coverage_agent = CoverageAgent(config.coverage)
        self.instruction_agent = InstructionAgent(config.audit)
        self.explainability = ScopeModelExplainability()

        self.group_df: pd.DataFrame | None = None
        self.findings_df: pd.DataFrame | None = None
        self.identified_risks: pd.DataFrame = pd.DataFrame(columns=RISK_COLUMNS)
        self.financial_status: dict[str, Any] = {
            "provided": False,
            "used": False,
            "message": "Financial data: risk discovery has not run.",
        }
        self.risk_result: RiskPredictionResult | None = None
        self.guardrail_coverage_status: GuardrailCoverageStatus | None = None
        self.human_review_artifacts: HumanReviewArtifacts | None = None
        self.coverage_result: CoverageResult | None = None
        self.instruction_result: InstructionResult | None = None
        self.crew_output: Any = None

        self.crew = self._build_crew()

    def run(self, financial_file: Path | None = None) -> dict[str, Any]:
        """Run the CrewAI-orchestrated ML audit workflow."""
        self.group_df, self.findings_df = self.data_loader.load_group_data(
            self.config.group_data_path,
            self.config.findings_data_path,
        )
        self._run_risk_discovery_stage(financial_file)

        self.crew_output = self.crew.kickoff(inputs={})
        export_paths = self._export_outputs()

        return {
            "crew_output": self.crew_output,
            "risk_result": self.risk_result,
            "coverage_result": self.coverage_result,
            "instruction_result": self.instruction_result,
            "export_paths": export_paths,
            "agent_outputs": self._agent_output_summary(),
            "identified_risks": self.identified_risks,
            "financial_status": self.financial_status,
        }

    def _run_risk_discovery_stage(self, financial_file: Path | None) -> None:
        if self.group_df is None or self.findings_df is None:
            raise RuntimeError("Risk discovery requires loaded group and findings data")
        client_data, financial_status = build_official_risk_discovery_input(
            self.group_df,
            self.findings_df,
            financial_file=financial_file,
        )
        self.identified_risks = discover_risks(client_data)
        self.financial_status = financial_status

    def _build_crew(self) -> Crew:
        llm = LocalCrewLLM()

        risk_agent = Agent(
            role="Risk Prediction Agent",
            goal="Analyze GASCO entities and collect local ML scope predictions.",
            backstory="A deterministic audit ML analyst that relies on MLScopeEngine outputs.",
            tools=[self.prediction_tool],
            llm=llm,
            verbose=False,
        )
        coverage_agent = Agent(
            role="Coverage Agent",
            goal="Evaluate audit coverage and risky uncovered components.",
            backstory="A group audit coverage specialist focused on sufficient asset coverage.",
            llm=llm,
            verbose=False,
        )
        instruction_agent = Agent(
            role="Instruction Agent",
            goal="Draft audit instructions and final narratives from ML recommendations.",
            backstory="A group audit instruction drafter that preserves ML explanations.",
            llm=llm,
            verbose=False,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="function callbacks cannot be serialized.*")
            warnings.filterwarnings("ignore", message="method callbacks cannot be serialized.*")
            risk_task = Task(
                name="Risk prediction",
                description="Run local ML predictions for all GASCO group entities.",
                expected_output="A table of predicted scopes, confidences, and explanations.",
                agent=risk_agent,
                callback=self._run_risk_prediction_stage,
            )
            coverage_task = Task(
                name="Coverage evaluation",
                description="Evaluate audit coverage from ML recommendations.",
                expected_output="Coverage summary and risky uncovered entities.",
                agent=coverage_agent,
                context=[risk_task],
                callback=self._run_coverage_stage,
            )
            instruction_task = Task(
                name="Instruction drafting",
                description="Generate audit instructions and final ML audit narrative.",
                expected_output="Audit instruction table and final narrative.",
                agent=instruction_agent,
                context=[risk_task, coverage_task],
                callback=self._run_instruction_stage,
            )

        return Crew(
            agents=[risk_agent, coverage_agent, instruction_agent],
            tasks=[risk_task, coverage_task, instruction_task],
            process=Process.sequential,
            verbose=False,
            memory=False,
            tracing=False,
        )

    def _run_risk_prediction_stage(self, _task_output: Any) -> None:
        if self.group_df is None:
            raise RuntimeError("Group data must be loaded before risk prediction")
        self.risk_result = self.risk_prediction_agent.analyze_entities(self.group_df)
        guarded_df, guardrail_coverage_status = self.bdo_guardrails.apply(
            self.risk_result.scoped_df,
            self.risk_result.feature_df,
            self.identified_risks,
        )
        self.guardrail_coverage_status = guardrail_coverage_status
        self.risk_result.scoped_df = self.bdo_methodology_mapper.map_outputs(guarded_df)
        self.human_review_artifacts = self.human_review_layer.prepare(
            self.risk_result.scoped_df,
            self.risk_result.explanations,
        )

    def _run_coverage_stage(self, _task_output: Any) -> None:
        if self.risk_result is None:
            raise RuntimeError("Risk prediction must run before coverage evaluation")
        self.coverage_result = self.coverage_agent.evaluate_coverage(
            self.risk_result.scoped_df
        )

    def _run_instruction_stage(self, _task_output: Any) -> None:
        if self.risk_result is None or self.coverage_result is None:
            raise RuntimeError("Risk prediction and coverage must run before instructions")
        self.instruction_result = self.instruction_agent.generate_instructions(
            self.risk_result.scoped_df,
            self.risk_result.explanations,
            self.coverage_result.summary,
        )

    def _export_outputs(self) -> dict[str, Path]:
        if (
            self.risk_result is None
            or self.coverage_result is None
            or self.instruction_result is None
            or self.human_review_artifacts is None
        ):
            raise RuntimeError("CrewAI workflow did not complete all stages")

        export_paths = {
            "scopes": self.csv_exporter.export_scope_recommendations(
                self.risk_result.scoped_df,
                "significance_scope_recommendation.csv",
            ),
            "coverage": self.csv_exporter.export_coverage_summary(
                self.coverage_result.summary,
                "coverage_summary.csv",
            ),
            "instructions": self.csv_exporter.export_instructions(
                self.instruction_result.instruction_df,
                "group_audit_instructions.csv",
            ),
            "risky_uncovered": self.output_directory / "risky_uncovered_entities.csv",
            "feature_importance": self.explainability.save_feature_importance(
                self.output_directory / "feature_importance.csv"
            ),
            "prediction_explanations": self.output_directory / "prediction_explanations.txt",
            "workflow_summary": self.output_directory / "crew_workflow_summary.txt",
            "bdo_documentation_memo": self.output_directory / "bdo_documentation_memo.txt",
            "component_instruction_pack": self.output_directory / "component_auditor_instruction_pack.csv",
        }
        export_paths.update(
            self.human_review_layer.export(
                self.human_review_artifacts,
                self.output_directory,
            )
        )

        self.coverage_result.risky_uncovered_df.to_csv(
            export_paths["risky_uncovered"],
            index=False,
        )
        self._write_prediction_explanations(export_paths["prediction_explanations"])
        self._write_workflow_summary(export_paths["workflow_summary"])
        self.bdo_documentation_memo.generate(
            self.risk_result.scoped_df,
            self.coverage_result.summary,
            export_paths["bdo_documentation_memo"],
        )
        self.component_instruction_pack.generate(
            self.risk_result.scoped_df,
            export_paths["component_instruction_pack"],
        )
        return export_paths

    def _write_prediction_explanations(self, output_path: Path) -> None:
        assert self.risk_result is not None
        preferred_entities = ["USA_Sub", "Germany_Sub", "Brazil_Sub"]
        selected_entities = [
            entity
            for entity in preferred_entities
            if entity in self.risk_result.explanations
        ]
        selected_entities.extend(
            entity
            for entity in self.risk_result.explanations
            if entity not in selected_entities
        )
        sections = [
            self.risk_result.explanations[entity]
            for entity in selected_entities[:3]
        ]
        output_path.write_text("\n\n".join(sections), encoding="utf-8")

    def _write_workflow_summary(self, output_path: Path) -> None:
        output_path.write_text(self._agent_output_summary(), encoding="utf-8")

    def _agent_output_summary(self) -> str:
        if self.risk_result is None or self.coverage_result is None or self.instruction_result is None:
            return "CrewAI workflow has not completed."

        return "\n".join([
            "GASCO Phase 3A CrewAI Workflow Summary",
            "",
            f"Risk Prediction Agent: {self.risk_result.summary}",
            f"Coverage Agent: {self.coverage_result.narrative}",
            self._guardrail_summary(),
            f"Instruction Agent: {self.instruction_result.final_narrative}",
            "",
            "CrewAI execution: sequential crew completed with local deterministic LLM callbacks.",
            "Prediction engine: MLScopeEngine using models/scope_model.pkl.",
            "Methodology guardrails: BDO-style post-ML validation applied before final exports.",
            "Financial-risk guardrails: deterministic RiskDiscoveryAgent outputs applied after ML prediction without retraining the model.",
            "Human-in-the-loop review: auditor workpaper and audit trail generated before final scoping approval.",
        ])

    def _guardrail_summary(self) -> str:
        assert self.risk_result is not None
        review_count = int(self.risk_result.scoped_df["Requires_Human_Review"].sum())
        adjusted_count = int(
            (
                self.risk_result.scoped_df["Original_ML_Scope"]
                != self.risk_result.scoped_df["Guardrail_Adjusted_Scope"]
            ).sum()
        )
        financial_guardrail_count = int(
            self.risk_result.scoped_df["Financial_Risk_Guardrail_Applied"].sum()
        )
        return (
            f"BDO Guardrails: {adjusted_count} recommendations adjusted; "
            f"{review_count} components require human review; "
            f"{financial_guardrail_count} financial-risk guardrail trigger(s) documented."
        )
