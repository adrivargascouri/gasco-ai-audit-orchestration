import pandas as pd


def load_group_data():
    """
    Component Data Ingestion Agent.
    Loads group structure data and prior audit findings.
    """

    # Load group structure JSON
    group_df = pd.read_json("data/group_structure.json")

    # Load findings CSV
    findings_df = pd.read_csv("data/findings_repo.csv")

    return group_df, findings_df


if __name__ == "__main__":
    group_df, findings_df = load_group_data()

    print("GROUP STRUCTURE DATA")
    print(group_df)

    print("\nFINDINGS REPOSITORY")
    print(findings_df)