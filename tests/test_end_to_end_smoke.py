from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .conftest import REPO_ROOT


def _copy_official_pipeline_project(destination: Path) -> Path:
    for directory_name in ["src", "modular", "data", "models"]:
        shutil.copytree(
            REPO_ROOT / directory_name,
            destination / directory_name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    return destination


def test_official_company_template_pipeline_smoke(tmp_path: Path) -> None:
    project_root = _copy_official_pipeline_project(tmp_path / "gasco_smoke")
    command = [
        sys.executable,
        "src/main_crewai.py",
        "--group-file",
        "data/templates/company_group_structure_template.csv",
        "--findings-file",
        "data/templates/company_findings_template.csv",
        "--financial-file",
        "data/templates/company_financial_data_template.csv",
    ]

    subprocess.run(
        command,
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    output_dir = project_root / "outputs_crewai"
    expected_outputs = [
        "significance_scope_recommendation.csv",
        "identified_risks.csv",
        "final_approved_scope.csv",
        "audit_trail.csv",
    ]

    for filename in expected_outputs:
        output_path = output_dir / filename
        assert output_path.is_file(), f"Missing pipeline output: {filename}"
        assert output_path.stat().st_size > 0, f"Empty pipeline output: {filename}"

    scope_df = pd.read_csv(output_dir / "significance_scope_recommendation.csv")
    assert "Financial_Risk_Guardrail_Action" in scope_df.columns
    assert scope_df["Financial_Risk_Guardrail_Applied"].any()
