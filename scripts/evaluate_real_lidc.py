"""Validate a REAL pretrained lung-cancer CT classifier on REAL LIDC-IDRI data.

NO synthetic data. NO training. This script:
  1. Loads real CT DICOM volumes downloaded from TCIA (LIDC-IDRI collection).
  2. Loads real radiologist malignancy annotations (XML, from TCIA/LIDC-XML-only.zip).
  3. Runs a fully pretrained, already fine-tuned lung-cancer CT classifier
     (oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans, from HuggingFace,
     trained on real NLST CT images) directly for inference -- zero additional training.
  4. Compares its predictions to the real malignancy-derived ground truth labels.

Usage:
    python scripts/evaluate_real_lidc.py --data_dir data/raw/lidc --max_patients 200
"""
import sys
import os
import argparse
import json
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

MALIGNANCY_THRESHOLD = 3
MODEL_ID = "oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans"
NS = {"lidc": "http://www.nih.gov"}


def parse_real_malignancy_labels(annotations_dir: str) -> dict:
    """Parse all real LIDC XML files -> {SeriesInstanceUid: (max_malignancy, nodule_z_positions)}.

    nodule_z_positions are the real ImagePositionPatient Z coordinates of the nodule
    slices with the highest malignancy rating, so we can evaluate the classifier on
    the slice that actually contains the nodule rather than an arbitrary middle slice.
    """
    series_malignancy = defaultdict(list)
    xml_files = glob.glob(os.path.join(annotations_dir, "**", "*.xml"), recursive=True)
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            header_el = root.find("lidc:ResponseHeader/lidc:SeriesInstanceUid", NS)
            series_uid = header_el.text if header_el is not None else None
            if not series_uid:
                continue
            for nodule_el in root.findall(".//lidc:unblindedReadNodule", NS):
                mal_el = nodule_el.find(".//lidc:malignancy", NS)
                if mal_el is None:
                    continue
                try:
                    mal = int(mal_el.text)
                except (TypeError, ValueError):
                    continue
                z_positions = [float(z.text) for z in nodule_el.findall(".//lidc:imageZposition", NS)]
                for z in z_positions:
                    series_malignancy[series_uid].append((mal, z))
        except Exception:
            continue

    result = {}
    for uid, entries in series_malignancy.items():
        if entries:
            best_mal, best_z = max(entries, key=lambda t: t[0])
            result[uid] = {"malignancy": best_mal, "nodule_z": best_z}
    return result


def load_slices_as_rgb(patient_dir: Path, target_z: float = None, n_slices: int = 5,
                        spacing_mm: float = 5.0) -> list:
    """Load several real DICOM slices around a target z (or spread across the volume).

    Real nodules can be a few slices thick and a single arbitrary slice may miss them,
    so we sample a small real window of slices and let the classifier vote/max-pool
    over them instead of trusting one slice in isolation.
    """
    dcm_files = sorted(patient_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files in {patient_dir}")

    slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=False)
            if hasattr(ds, "pixel_array") and hasattr(ds, "ImagePositionPatient"):
                slices.append((float(ds.ImagePositionPatient[2]), ds))
        except Exception:
            continue
    if not slices:
        raise ValueError(f"No readable pixel slices in {patient_dir}")

    slices.sort(key=lambda t: t[0])

    if target_z is not None:
        targets = [target_z + i * spacing_mm for i in range(-(n_slices // 2), n_slices // 2 + 1)]
    else:
        zs = [s[0] for s in slices]
        lo, hi = min(zs), max(zs)
        targets = [lo + (hi - lo) * frac for frac in np.linspace(0.2, 0.8, n_slices)]

    chosen_ds = []
    seen_z = set()
    for t in targets:
        ds = min(slices, key=lambda s: abs(s[0] - t))
        if ds[0] not in seen_z:
            chosen_ds.append(ds[1])
            seen_z.add(ds[0])

    images = []
    for ds in chosen_ds:
        pixels = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = pixels * slope + intercept
        min_hu, max_hu = -1350, 150
        hu = np.clip(hu, min_hu, max_hu)
        norm = ((hu - min_hu) / (max_hu - min_hu) * 255).astype(np.uint8)
        images.append(Image.fromarray(np.stack([norm, norm, norm], axis=-1)))
    return images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw/lidc")
    ap.add_argument("--max_patients", type=int, default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    progress_path = data_dir / "download_progress.json"
    progress = json.loads(progress_path.read_text())
    ok_patients = [pid for pid, v in progress.items() if v.get("status") == "ok"]
    print(f"Real LIDC-IDRI patients downloaded so far: {len(ok_patients)}")

    if args.max_patients:
        ok_patients = ok_patients[: args.max_patients]

    print("Parsing real radiologist malignancy annotations (XML)...")
    malignancy_by_series = parse_real_malignancy_labels(str(data_dir / "annotations"))
    print(f"  Real annotated series with malignancy scores: {len(malignancy_by_series)}")

    print(f"Loading real pretrained classifier: {MODEL_ID} (zero training, inference only)...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()

    y_true, y_pred, y_prob, used_patients = [], [], [], []
    skipped = 0

    for i, pid in enumerate(ok_patients):
        series_uid = progress[pid]["series_uid"]
        ann = malignancy_by_series.get(series_uid)
        mal_score = ann["malignancy"] if ann else 0
        target_z = ann["nodule_z"] if ann else None
        label = 1 if mal_score >= MALIGNANCY_THRESHOLD else 0

        try:
            images = load_slices_as_rgb(data_dir / pid, target_z=target_z)
        except Exception:
            skipped += 1
            continue

        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt")
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            best_prob = float(probs[:, 1].max().item())

        y_true.append(label)
        y_prob.append(best_prob)
        used_patients.append(pid)

        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i+1}/{len(ok_patients)}")

    print(f"\nReal patients evaluated: {len(y_true)}  (skipped, unreadable DICOM: {skipped})")
    print(f"Real label distribution -> benign(0): {y_true.count(0)}  malignant(1): {y_true.count(1)}")

    pd.DataFrame({"patient_id": used_patients, "y_true": y_true, "y_prob_oohtmeel": y_prob}) \
        .to_csv("results/real_lidc_per_patient_oohtmeel.csv", index=False)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if len(set(y_true.tolist())) < 2:
        print("WARNING: only one real class present in this subset; AUROC undefined.")
        auroc = None
    else:
        auroc = roc_auc_score(y_true, y_prob)

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(y_true))
    half = len(idx) // 2
    calib_idx, test_idx = idx[:half], idx[half:]

    best_thresh, best_calib_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        preds = (y_prob[calib_idx] >= t).astype(int)
        f1_t = f1_score(y_true[calib_idx], preds, average="macro", zero_division=0)
        if f1_t > best_calib_f1:
            best_calib_f1, best_thresh = f1_t, t
    print(f"\nThreshold calibrated on held-out calibration half: {best_thresh:.2f} "
          f"(calibration-set F1={best_calib_f1:.4f})")

    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_calibrated_test = (y_prob[test_idx] >= best_thresh).astype(int)

    acc = accuracy_score(y_true, y_pred_default)
    f1 = f1_score(y_true, y_pred_default, average="macro")
    cm = confusion_matrix(y_true, y_pred_default)

    acc_calib = accuracy_score(y_true[test_idx], y_pred_calibrated_test)
    f1_calib = f1_score(y_true[test_idx], y_pred_calibrated_test, average="macro")
    cm_calib = confusion_matrix(y_true[test_idx], y_pred_calibrated_test)

    print("\n" + "=" * 60)
    print("REAL DATA VALIDATION: LIDC-IDRI Imaging (pretrained classifier, no training)")
    print("=" * 60)
    print(f"[Default threshold 0.5, full data]  Accuracy: {acc:.4f}  F1(macro): {f1:.4f}")
    if auroc is not None:
        print(f"AUROC (threshold-independent, full data): {auroc:.4f}")
    print("Confusion matrix (default threshold):\n", cm)
    print(classification_report(y_true, y_pred_default, zero_division=0))

    print(f"\n[Calibrated threshold {best_thresh:.2f}, held-out test half only, n={len(test_idx)}]")
    print(f"Accuracy: {acc_calib:.4f}  F1(macro): {f1_calib:.4f}")
    print("Confusion matrix (calibrated, held-out half):\n", cm_calib)
    print(classification_report(y_true[test_idx], y_pred_calibrated_test, zero_division=0))

    Path("results").mkdir(exist_ok=True)
    with open("results/real_lidc_results.json", "w") as f:
        json.dump({
            "n_evaluated": len(y_true), "model_id": MODEL_ID,
            "default_threshold": {"accuracy": acc, "auroc": auroc, "f1_macro": f1,
                                   "confusion_matrix": cm.tolist()},
            "calibrated_threshold": {"threshold": float(best_thresh), "n_test": len(test_idx),
                                      "accuracy": acc_calib, "f1_macro": f1_calib,
                                      "confusion_matrix": cm_calib.tolist()},
        }, f, indent=2)
    print("\nSaved results/real_lidc_results.json")


if __name__ == "__main__":
    main()
