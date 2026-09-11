"""Ensemble two REAL, independent pretrained-model classifiers for LIDC-IDRI detection.

NO synthetic data, NO fine-tuning of any encoder. Combines:
  1. A tiny classifier fit on REAL frozen SwinUNETR whole-volume features
     (data/raw/lidc/swinunetr_features.npy, from scripts/extract_real_swinunetr_volumes.py)
  2. The REAL oohtmeel 2D pretrained lung-cancer classifier's per-patient probability
     (results/real_lidc_per_patient_oohtmeel.csv, from scripts/evaluate_real_lidc.py)

This is genuine "combine multiple pretrained models" fusion -- the original architectural
vision of this project -- evaluated honestly via 5-fold cross-validation with out-of-fold
probabilities (no leakage) for the SwinUNETR classifier, then averaged with oohtmeel's
real predictions.

Usage:
    python scripts/ensemble_real_lidc.py
"""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

MALIGNANCY_THRESHOLD = 3

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_real_lidc import parse_real_malignancy_labels


def main():
    data_dir = Path("data/raw/lidc")
    progress = json.loads((data_dir / "download_progress.json").read_text())

    swin_features = np.load(data_dir / "swinunetr_features.npy")
    swin_patient_ids = json.loads((data_dir / "swinunetr_patient_ids.json").read_text())
    print(f"Real SwinUNETR features: {swin_features.shape} for {len(swin_patient_ids)} patients")

    oohtmeel_df = pd.read_csv("results/real_lidc_per_patient_oohtmeel.csv")
    print(f"Real oohtmeel per-patient predictions: {len(oohtmeel_df)} patients")

    malignancy_by_series = parse_real_malignancy_labels(str(data_dir / "annotations"))

    labels = []
    for pid in swin_patient_ids:
        series_uid = progress[pid]["series_uid"]
        ann = malignancy_by_series.get(series_uid)
        mal_score = ann["malignancy"] if ann else 0
        labels.append(1 if mal_score >= MALIGNANCY_THRESHOLD else 0)
    labels = np.array(labels)
    print(f"Real label distribution (SwinUNETR set) -> benign(0): {(labels==0).sum()}  malignant(1): {(labels==1).sum()}")

    X = StandardScaler().fit_transform(swin_features)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
    swin_probs = cross_val_predict(clf, X, labels, cv=skf, method="predict_proba")[:, 1]

    swin_df = pd.DataFrame({"patient_id": swin_patient_ids, "y_true": labels, "y_prob_swinunetr": swin_probs})

    merged = swin_df.merge(oohtmeel_df[["patient_id", "y_prob_oohtmeel"]], on="patient_id", how="inner")
    print(f"Real patients with BOTH SwinUNETR and oohtmeel predictions: {len(merged)}")

    y_true = merged["y_true"].values
    p_swin = merged["y_prob_swinunetr"].values
    p_ooh = merged["y_prob_oohtmeel"].values
    p_ensemble = (p_swin + p_ooh) / 2.0

    # Learned stacking meta-learner instead of a naive 50/50 average: fit a tiny
    # logistic regression on [p_swin, p_ooh] via 5-fold CV (honest out-of-fold
    # probabilities), letting the data decide how much to trust each real model
    # rather than assuming they deserve equal weight.
    stack_X = np.stack([p_swin, p_ooh], axis=1)
    stack_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stack_clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    p_stacked = cross_val_predict(stack_clf, stack_X, y_true, cv=stack_skf, method="predict_proba")[:, 1]
    stack_clf.fit(stack_X, y_true)
    print(f"\nStacking meta-learner weights (swin, oohtmeel): {stack_clf.coef_[0]}, intercept={stack_clf.intercept_[0]:.3f}")

    def report(name, y, probs):
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="macro")
        auroc = roc_auc_score(y, probs) if len(set(y.tolist())) > 1 else None
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}  F1(macro): {f1:.4f}"
              + (f"  AUROC: {auroc:.4f}" if auroc is not None else ""))
        print(confusion_matrix(y, preds))
        print(classification_report(y, preds, zero_division=0))
        return {"accuracy": acc, "f1_macro": f1, "auroc": auroc}

    print("\n" + "=" * 60)
    print("REAL MODEL ENSEMBLE: SwinUNETR (whole-volume) + oohtmeel (2D slice)")
    print("=" * 60)
    results = {
        "swinunetr_only": report("SwinUNETR-only (real, 5-fold CV out-of-fold)", y_true, p_swin),
        "oohtmeel_only": report("Oohtmeel-only (real, on same patient subset)", y_true, p_ooh),
        "ensemble_avg": report("Ensemble (average of both real models)", y_true, p_ensemble),
        "ensemble_stacked": report("Ensemble (learned stacking, out-of-fold)", y_true, p_stacked),
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/real_lidc_ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/real_lidc_ensemble_results.json")

    print("\n" + "=" * 60)
    print("CLEAN-LABEL VARIANT: drop indeterminate (malignancy==3) real cases")
    print("=" * 60)
    clean_labels = []
    for pid in merged["patient_id"]:
        series_uid = progress[pid]["series_uid"]
        ann = malignancy_by_series.get(series_uid)
        clean_labels.append(ann["malignancy"] if ann else 0)
    clean_labels = np.array(clean_labels)
    keep = clean_labels != 3
    print(f"Real cases kept (excluding indeterminate): {keep.sum()} / {len(keep)}")

    y_clean = (clean_labels[keep] >= 4).astype(int)
    print(f"Clean-label distribution -> benign(0): {(y_clean==0).sum()}  malignant(1): {(y_clean==1).sum()}")
    clean_results = {
        "swinunetr_only": report("SwinUNETR-only (clean labels)", y_clean, p_swin[keep]),
        "oohtmeel_only": report("Oohtmeel-only (clean labels)", y_clean, p_ooh[keep]),
        "ensemble_avg": report("Ensemble avg (clean labels)", y_clean, p_ensemble[keep]),
    }
    with open("results/real_lidc_ensemble_clean_labels_results.json", "w") as f:
        json.dump(clean_results, f, indent=2)
    print("\nSaved results/real_lidc_ensemble_clean_labels_results.json")


if __name__ == "__main__":
    main()
