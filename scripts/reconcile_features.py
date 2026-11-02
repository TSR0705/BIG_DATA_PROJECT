"""
Feature Family Reconciliation Script for Phase 10 Audit.
Maps all 256 model features across the 7 disjoint families and explains the structure.
"""
import os
import yaml
import pandas as pd

def reconcile():
    features_path = "configs/features.yaml"
    with open(features_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Exclude SK_ID_CURR and TARGET
    model_features = {k: v for k, v in cfg.items() if k not in ["SK_ID_CURR", "TARGET"]}
    
    rows = []
    family_counts = {}
    source_counts = {}
    ablation_counts = {}

    for feat_name, meta in model_features.items():
        family = meta.get("family", "UNKNOWN")
        source = meta.get("source", "UNKNOWN")
        excluded = meta.get("excluded_from_model", False)
        leakage = meta.get("leakage_status", "SAFE")
        desc = meta.get("description", meta.get("definition", ""))
        
        # Determine derived / cross-domain status
        if source == "cross_domain" or family == "DERIVED":
            derived_status = "Cross-Domain Interaction (Multi-Table)"
            ablation_group = "derived"
        elif feat_name.startswith("APP_") or feat_name.startswith("FLAG_"):
            if feat_name in ["FLAG_DAYS_EMPLOYED_SENTINEL", "FLAG_CODE_GENDER_XNA", "FLAG_ORGANIZATION_TYPE_XNA"]:
                derived_status = "Data-Quality Sentinel Flag"
            elif any(sub in feat_name for sub in ["RATIO", "MEAN", "PROD", "PERCENT", "PER_PERSON", "PER_CHILD", "RANGE", "COUNT", "YEARS"]):
                derived_status = "Intra-Application Engineered Feature"
            else:
                derived_status = "Original Application Feature"
            ablation_group = "app"
        elif family == "APP":
            derived_status = "Original Cleaned Attribute"
            ablation_group = "app"
        elif family in ["BURO", "BURO_BAL"]:
            if family == "BURO_BAL":
                derived_status = "Bureau Balance Aggregated Rollup"
            else:
                derived_status = "Bureau Aggregated Rollup"
            ablation_group = "bureau"
        elif family == "PREV":
            derived_status = "Previous Application Aggregated Rollup"
            ablation_group = "prev"
        elif family == "INST":
            derived_status = "Installments Payment Aggregated Rollup"
            ablation_group = "installments"
        elif family == "CC":
            derived_status = "Credit Card Balance Aggregated Rollup"
            ablation_group = "credit_card"
        elif family == "POS":
            derived_status = "POS/CASH Balance Aggregated Rollup"
            ablation_group = "pos_cash"
        else:
            derived_status = "Standard Feature"
            ablation_group = family.lower()

        # Inclusion / exclusion reason
        inclusion_reason = f"Included in model feature set; leakage={leakage}; excluded={excluded}; provides signal from {source}."

        rows.append({
            "feature_name": feat_name,
            "source_table": source,
            "feature_family": family,
            "derived_status": derived_status,
            "ablation_group": ablation_group,
            "inclusion_exclusion_reason": inclusion_reason
        })

        family_counts[family] = family_counts.get(family, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
        ablation_counts[ablation_group] = ablation_counts.get(ablation_group, 0) + 1

    df = pd.DataFrame(rows)
    os.makedirs("reports", exist_ok=True)
    out_csv = "reports/feature_family_reconciliation.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} feature mappings to {out_csv}")
    print("\n--- Summary by Feature Family in YAML ---")
    for k, v in sorted(family_counts.items()):
        print(f"  {k}: {v}")
    print("\n--- Summary by Ablation Group in E1 ---")
    for k, v in sorted(ablation_counts.items()):
        print(f"  {k}: {v}")
    print("\n--- Summary by Source Table ---")
    for k, v in sorted(source_counts.items()):
        print(f"  {k}: {v}")
    print(f"\nTotal Features: {len(df)} (Matches expected 256: {len(df) == 256})")

if __name__ == "__main__":
    reconcile()
