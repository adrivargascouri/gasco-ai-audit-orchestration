import pandas as pd


def recommend_scope(group_df):
    """
    Significance & Scope Recommendation Agent.
    Decides the audit scope for each component based on assets and risk level.
    """

    total_assets = group_df["Assets"].sum()

    def classify_component(row):
        asset_percentage = row["Assets"] / total_assets

        if asset_percentage > 0.15:
            return "Full Scope"
        elif row["Risk_Level"] == "High":
            return "Specific Procedures"
        else:
            return "Analytical Procedures"

    group_df["Asset_Percentage"] = group_df["Assets"] / total_assets
    group_df["Asset_Percentage"] = group_df["Asset_Percentage"] * 100
    group_df["Recommended_Scope"] = group_df.apply(classify_component, axis=1)

    return group_df


if __name__ == "__main__":
    group_df = pd.read_json("data/group_structure.json")

    result_df = recommend_scope(group_df)

    print("SIGNIFICANCE & SCOPE RECOMMENDATION")
    print(result_df[["Entity", "Country", "Assets", "Asset_Percentage", "Risk_Level", "Recommended_Scope"]])