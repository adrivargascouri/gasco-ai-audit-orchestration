import pandas as pd
from significance_agent import recommend_scope


def generate_instructions(group_df):
    """
    Instruction Drafting Agent.
    Generates audit instructions for each component based on the recommended scope.
    """

    scoped_df = recommend_scope(group_df)

    instructions = []

    for _, row in scoped_df.iterrows():
        entity = row["Entity"]
        country = row["Country"]
        risk_level = row["Risk_Level"]
        scope = row["Recommended_Scope"]

        if scope == "Full Scope":
            instruction = (
                f"For {entity} in {country}, perform full scope audit procedures. "
                f"The component should be audited comprehensively, including financial statements, "
                f"key balances, revenue, expenses, assets, liabilities, and internal controls."
            )

        elif scope == "Specific Procedures":
            instruction = (
                f"For {entity} in {country}, perform specific audit procedures focused on high-risk areas. "
                f"Pay special attention to significant account balances, unusual transactions, "
                f"management estimates, and areas with increased risk."
            )

        else:
            instruction = (
                f"For {entity} in {country}, perform analytical procedures only. "
                f"Review financial trends, compare current results with prior periods, "
                f"and investigate any unusual fluctuations."
            )

        instructions.append({
            "Entity": entity,
            "Country": country,
            "Risk_Level": risk_level,
            "Recommended_Scope": scope,
            "Audit_Instruction": instruction
        })

    return pd.DataFrame(instructions)


if __name__ == "__main__":
    group_df = pd.read_json("data/group_structure.json")

    instruction_df = generate_instructions(group_df)

    print("GROUP AUDIT INSTRUCTIONS")
    print(instruction_df[["Entity", "Country", "Risk_Level", "Recommended_Scope", "Audit_Instruction"]])