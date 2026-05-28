"""Generate component auditor instruction pack exports."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class ComponentInstructionPack:
    """Write a CSV instruction pack for component auditor communication."""

    COLUMNS = [
        "Entity",
        "Country",
        "Original_ML_Scope",
        "Guardrail_Adjusted_Scope",
        "ML_Confidence",
        "Risk_Response_Level",
        "Component_Selected_For_Work",
        "Component_Auditor_Involvement",
        "Basis_For_Selection",
        "Guardrail_Reason",
        "Requires_Human_Review",
        "Expected_Component_Auditor_Communication",
    ]

    def generate(self, scoped_df: pd.DataFrame, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pack_df = scoped_df.copy()
        pack_df["ML_Confidence"] = pack_df["Prediction_Confidence"]
        pack_df["Expected_Component_Auditor_Communication"] = pack_df.apply(
            self._expected_communication,
            axis=1,
        )
        pack_df[self.COLUMNS].to_csv(output_path, index=False)
        return output_path

    def _expected_communication(self, row: pd.Series) -> str:
        if row["Component_Auditor_Involvement"] == "Not Required":
            return "No component auditor work request planned unless the group team revises scoping."
        if bool(row["Requires_Human_Review"]):
            return "Confirm proposed scope, planned procedures, timing, reporting format, and matters needing group team review."
        return "Confirm planned procedures, reporting timetable, and significant matters identified during component work."
