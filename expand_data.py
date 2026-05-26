#!/usr/bin/env python3
"""Expand historical audit data from 138 to 200+ rows while maintaining realistic patterns."""

import pandas as pd
import numpy as np
from pathlib import Path


def expand_dataset(input_path, output_path, target_rows=200):
    """Expand dataset while maintaining realistic audit patterns."""

    # Load existing data
    df = pd.read_csv(input_path)
    print(f"Original dataset: {len(df)} rows")

    # Analyze current distribution
    target_dist = df['target_scope'].value_counts()
    print(f"\nCurrent target_scope distribution:\n{target_dist}\n")

    # Determine target distribution (approximately)
    # Typically: Analytical > Specific > Full
    current_pct = target_dist / len(df)
    rows_to_add = target_rows - len(df)

    # Create new rows by variation of existing patterns
    new_rows = []

    # Get unique countries and their risk scores
    countries = df['country'].unique()
    country_risk_map = df.groupby('country')['country_risk_score'].first().to_dict()

    # Parameters for realistic generation
    np.random.seed(42)

    # Generate rows maintaining target_scope distribution
    target_counts = target_dist.to_dict()
    target_goal = {
        'Analytical Procedures': int(target_rows * 0.65),
        'Specific Procedures': int(target_rows * 0.25),
        'Full Scope': int(target_rows * 0.10)
    }

    print(f"Target distribution goal:\n{target_goal}\n")

    # Generate new entities
    entity_counter = 1000

    for scope in ['Analytical Procedures', 'Specific Procedures', 'Full Scope']:
        current_count = target_counts.get(scope, 0)
        needed = target_goal[scope] - current_count

        if needed <= 0:
            continue

        # Sample reference rows for this scope
        scope_examples = df[df['target_scope'] == scope]

        for _ in range(needed):
            # Pick random example from this scope
            example = scope_examples.sample(1).iloc[0]

            # Create variation
            new_row = example.copy()

            # Vary entity name and some features
            new_row['entity'] = f"{example['entity']}_V{entity_counter}"
            entity_counter += 1

            # Vary numeric features by ±10-20%
            noise_pct = np.random.uniform(-0.2, 0.2)
            new_row['revenue'] = int(example['revenue'] * (1 + noise_pct * 0.5))
            new_row['assets'] = int(example['assets'] * (1 + noise_pct * 0.5))

            # Recalculate percentages (assuming total is sum of all)
            total_rev = df['revenue'].sum()
            total_assets = df['assets'].sum()
            new_row['revenue_percentage'] = round(new_row['revenue'] / (total_rev + new_row['revenue']) * 100, 1)
            new_row['assets_percentage'] = round(new_row['assets'] / (total_assets + new_row['assets']) * 100, 1)

            # Vary growth rate and liquidity
            new_row['growth_rate'] = round(example['growth_rate'] + np.random.uniform(-1, 1), 1)
            new_row['growth_rate'] = max(0, new_row['growth_rate'])  # Can't be negative
            new_row['liquidity_ratio'] = round(example['liquidity_ratio'] + np.random.uniform(-0.1, 0.1), 2)
            new_row['liquidity_ratio'] = max(1.0, new_row['liquidity_ratio'])  # Reasonable min

            # Small variations in findings
            new_row['prior_findings_count'] = max(0, example['prior_findings_count'] + np.random.randint(-1, 2))
            new_row['severe_findings_count'] = max(0, example['severe_findings_count'] + np.random.randint(-1, 1))

            new_rows.append(new_row)

    # Combine original and new data
    new_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([df, new_df], ignore_index=True)

    # Verify no duplicates in entity names (just in case)
    duplicates = combined_df['entity'].duplicated().sum()
    if duplicates > 0:
        print(f"Warning: {duplicates} duplicate entity names detected")

    # Verify all required columns present and no NaN
    required_cols = {
        'entity', 'country', 'revenue', 'assets', 'revenue_percentage', 'assets_percentage',
        'risk_level', 'country_risk_score', 'prior_findings_count', 'severe_findings_count',
        'growth_rate', 'liquidity_ratio', 'manual_risk_flag', 'target_scope'
    }

    missing_cols = required_cols - set(combined_df.columns)
    if missing_cols:
        print(f"Error: Missing columns {missing_cols}")
        return None

    # Check for missing values
    missing_count = combined_df[list(required_cols)].isnull().sum().sum()
    if missing_count > 0:
        print(f"Error: {missing_count} missing values detected")
        return None

    # Save expanded dataset
    combined_df.to_csv(output_path, index=False)

    # Print statistics
    print(f"\n✓ Expanded dataset saved: {output_path}")
    print(f"Final dataset: {len(combined_df)} rows (added {len(combined_df) - len(df)} rows)")
    print(f"\nFinal target_scope distribution:")
    final_dist = combined_df['target_scope'].value_counts().sort_index()
    for scope, count in final_dist.items():
        pct = count / len(combined_df) * 100
        print(f"  {scope}: {count} ({pct:.1f}%)")

    print(f"\nNo missing values: {missing_count == 0}")

    return combined_df


if __name__ == "__main__":
    input_path = Path("data/raw/historical_audit_data.csv")
    output_path = Path("data/raw/historical_audit_data.csv")

    expand_dataset(input_path, output_path, target_rows=200)
