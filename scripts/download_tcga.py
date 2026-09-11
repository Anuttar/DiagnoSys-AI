"""Download TCGA-LUAD and TCGA-LUSC clinical data from GDC portal.

Downloads open-access clinical metadata (demographics, staging, smoking history)
and pathology report file IDs via the GDC REST API. No authentication required.

Usage:
    python scripts/download_tcga.py
    python scripts/download_tcga.py --include_reports   # also download pathology PDFs
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import requests
except ImportError:
    print("Install requests: python -m pip install requests")
    sys.exit(1)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

GDC_API = "https://api.gdc.cancer.gov"
PROJECTS = ["TCGA-LUAD", "TCGA-LUSC"]


def download_clinical_data(output_dir: str):
    """Download clinical metadata for TCGA-LUAD and TCGA-LUSC."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filters = {
        "op": "in",
        "content": {
            "field": "project.project_id",
            "value": PROJECTS
        }
    }

    params = {
        "filters": json.dumps(filters),
        "expand": "diagnoses,demographic,exposures,project",
        "size": 2000,
        "format": "JSON",
    }

    print("Querying GDC API for clinical data...")
    try:
        resp = requests.get(f"{GDC_API}/cases", params=params, timeout=60, verify=False)
        resp.raise_for_status()
    except requests.exceptions.SSLError:
        print("SSL error — trying without verification...")
        resp = requests.get(f"{GDC_API}/cases", params=params, timeout=60, verify=False)
        resp.raise_for_status()

    data = resp.json()
    cases = data.get("data", {}).get("hits", [])
    print(f"Retrieved {len(cases)} cases")

    with open(output_path / "clinical.json", "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Saved clinical.json ({len(cases)} cases)")

    import pandas as pd
    records = []
    for case in cases:
        demos = case.get("demographic", {}) or {}
        diags = (case.get("diagnoses") or [{}])[0]
        expos = (case.get("exposures") or [{}])[0]
        project = case.get("project", {}) or {}

        records.append({
            "case_id": case.get("case_id", ""),
            "project": project.get("project_id", ""),
            "primary_diagnosis": diags.get("primary_diagnosis", ""),
            "age_at_diagnosis": diags.get("age_at_diagnosis"),
            "ajcc_pathologic_stage": diags.get("ajcc_pathologic_stage", ""),
            "ajcc_pathologic_t": diags.get("ajcc_pathologic_t", ""),
            "ajcc_pathologic_n": diags.get("ajcc_pathologic_n", ""),
            "ajcc_pathologic_m": diags.get("ajcc_pathologic_m", ""),
            "tumor_grade": diags.get("tumor_grade", ""),
            "morphology": diags.get("morphology", ""),
            "site_of_resection": diags.get("site_of_resection_or_biopsy", ""),
            "gender": demos.get("sex_at_birth", demos.get("gender", "")),
            "race": demos.get("race", ""),
            "ethnicity": demos.get("ethnicity", ""),
            "vital_status": demos.get("vital_status", ""),
            "days_to_death": demos.get("days_to_death"),
            "tobacco_smoking_status": expos.get("tobacco_smoking_status", ""),
            "pack_years_smoked": expos.get("pack_years_smoked"),
            "alcohol_history": expos.get("alcohol_history", ""),
            "bmi": expos.get("bmi"),
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path / "clinical.csv", index=False)
    print("Saved clinical.csv")

    print("\nDataset Summary:")
    print(f"  Total cases: {len(df)}")
    for proj in PROJECTS:
        n = len(df[df["project"] == proj])
        print(f"  {proj}: {n} cases")
    print(f"  Stages: {df['ajcc_pathologic_stage'].value_counts().to_dict()}")
    print(f"  Gender: {df['gender'].value_counts().to_dict()}")

    return df


def download_pathology_reports(output_dir: str, max_reports: int = 50):
    """Download pathology report PDFs from GDC."""
    output_path = Path(output_dir) / "pathology_reports"
    output_path.mkdir(parents=True, exist_ok=True)

    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": PROJECTS}},
            {"op": "=", "content": {"field": "data_type", "value": "Pathology Report"}},
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.case_id",
        "size": max_reports,
        "format": "JSON",
    }

    print(f"\nQuerying GDC for pathology reports (max {max_reports})...")
    resp = requests.get(f"{GDC_API}/files", params=params, timeout=60, verify=False)
    resp.raise_for_status()

    files = resp.json().get("data", {}).get("hits", [])
    print(f"Found {len(files)} pathology reports")

    downloaded = 0
    for f in files:
        file_id = f["file_id"]
        file_name = f.get("file_name", f"{file_id}.pdf")
        save_path = output_path / file_name

        if save_path.exists():
            downloaded += 1
            continue

        try:
            r = requests.get(f"{GDC_API}/data/{file_id}", timeout=120, verify=False)
            r.raise_for_status()
            with open(save_path, "wb") as fp:
                fp.write(r.content)
            downloaded += 1
            if downloaded % 10 == 0:
                print(f"  Downloaded {downloaded}/{len(files)}")
        except Exception as e:
            print(f"  Failed {file_name}: {e}")

    print(f"Downloaded {downloaded} pathology reports to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download TCGA lung cancer data")
    parser.add_argument("--output_dir", default="data/raw/tcga")
    parser.add_argument("--include_reports", action="store_true",
                        help="Also download pathology report PDFs")
    parser.add_argument("--max_reports", type=int, default=100)
    args = parser.parse_args()

    df = download_clinical_data(args.output_dir)

    if args.include_reports:
        download_pathology_reports(args.output_dir, args.max_reports)

    print(f"\nTCGA data saved to {args.output_dir}/")
    print("Next steps:")
    print("  1. Run: python scripts/extract_features.py --source tcga")
    print("  2. Then retrain with real data")


if __name__ == "__main__":
    main()
