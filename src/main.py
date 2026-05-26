import os
import pandas as pd
from data_ingestion import load_group_data

from significance_agent import recommend_scope
from coverage_agent import calculate_coverage
from instruction_agent import generate_instructions


def run_group_audit_orchestrator():
    """
    Main Orchestrator Agent.
    Coordinates all group audit agents.
    """

    # 1. Load group structure data
    group_df, findings_df = load_group_data()

    # 2. Run significance and scope recommendation agent
    scoped_df = recommend_scope(group_df)

    # 3. Run coverage aggregation agent
    coverage_results = calculate_coverage(group_df)

    # 4. Run instruction drafting agent
    instruction_df = generate_instructions(group_df)

    # 5. Create outputs folder if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    # 6. Export results to CSV files 
    scoped_df.to_csv("outputs/significance_scope_recommendation.csv", index=False)
    instruction_df.to_csv("outputs/group_audit_instructions.csv", index=False)
    
    coverage_summary = pd.DataFrame([{
        "Total_Assets": coverage_results["Total_Assets"],
        "Covered_Assets": coverage_results["Covered_Assets"],
        "Coverage_Percentage": coverage_results["Coverage_Percentage"],
        "Status": "Sufficient audit coverage"
        if coverage_results["Coverage_Percentage"] >= 70
         else "Warning: Coverage is too low"
    }])

    coverage_summary.to_csv("outputs/coverage_summary.csv", index=False)

    # 7. Display results
    print("\n==============================")
    print("GASCO GROUP AUDIT ORCHESTRATOR")
    print("==============================")

    print("\n1. SIGNIFICANCE & SCOPE RECOMMENDATION")
    print(scoped_df[["Entity", "Country", "Assets", "Risk_Level", "Recommended_Scope"]])

    print("\n2. GROUP AUDIT COVERAGE")
    print(f"Total Group Assets: ${coverage_results['Total_Assets']:,}")
    print(f"Covered Assets: ${coverage_results['Covered_Assets']:,}")
    print(f"Coverage Percentage: {coverage_results['Coverage_Percentage']:.2f}%")

    if coverage_results["Coverage_Percentage"] >= 70:
        print("Status: Sufficient audit coverage")
    else:
        print("Warning: Coverage is too low")

    print("\n3. GROUP AUDIT INSTRUCTIONS")
    pd.set_option("display.max_colwidth", None)
    print(instruction_df[["Entity", "Country", "Recommended_Scope", "Audit_Instruction"]])


if __name__ == "__main__":
    run_group_audit_orchestrator()