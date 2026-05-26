import pandas as pd
from significance_agent import recommend_scope


def calculate_coverage(group_df):
    """
    Coverage Aggregation Agent.
    Calculates total audit coverage for the group audit.
    """

    # Get scope recommendations
    scoped_df = recommend_scope(group_df)

    # Total group assets
    total_assets = scoped_df["Assets"].sum()

    # Components included in audit coverage
    covered_components = scoped_df[
        scoped_df["Recommended_Scope"].isin(
            ["Full Scope", "Specific Procedures"]
        )
    ]

    # Covered assets
    covered_assets = covered_components["Assets"].sum()

    # Coverage percentage
    coverage_percentage = (covered_assets / total_assets) * 100

    return {
        "Total_Assets": total_assets,
        "Covered_Assets": covered_assets,
        "Coverage_Percentage": coverage_percentage,
        "Covered_Components": covered_components
    }


if __name__ == "__main__":
    group_df = pd.read_json("data/group_structure.json")

    coverage_results = calculate_coverage(group_df)

    print("GROUP AUDIT COVERAGE RESULTS")
    print(f"Total Group Assets: ${coverage_results['Total_Assets']:,}")
    print(f"Covered Assets: ${coverage_results['Covered_Assets']:,}")
    print(f"Coverage Percentage: {coverage_results['Coverage_Percentage']:.2f}%")

    if coverage_results["Coverage_Percentage"] >= 70:
        print("STATUS: Sufficient audit coverage")
    else:
        print("WARNING: Coverage is too low")
        
    print("\nCovered Components:")
    print(
        coverage_results["Covered_Components"][
            ["Entity", "Country", "Assets", "Risk_Level", "Recommended_Scope"]
        ]
    )


