"""Extract REAL SwinUNETR features from REAL LIDC-IDRI CT volumes (whole-volume, as per
the original DiagnoSys-AI architecture design), for genuine multimodal-signal improvement.

NO synthetic data. Uses the existing LIDCLoader (src/data/lidc_loader.py) to read real
DICOM series via SimpleITK, resample to a fixed volume, and apply the same real lung
windowing/normalization used elsewhere in this project. Features come from the frozen,
pretrained MONAI SwinUNETR encoder (same weights already used throughout this project) --
never fine-tuned.

This lets us fit a SECOND real, independent pretrained-model-based classifier for LIDC
detection and ensemble it with the oohtmeel 2D slice classifier
(scripts/evaluate_real_lidc.py) -- true "combine multiple pretrained models" fusion,
which is the original architectural intent of this project.

Usage:
    python scripts/extract_real_swinunetr_volumes.py --data_dir data/raw/lidc
"""
import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from src.data.lidc_loader import LIDCLoader

PROJECT_ROOT = Path(os.path.dirname(__file__)).parent
SWINUNETR_WEIGHTS = PROJECT_ROOT / "models" / "pretrained" / "model_swinvit.pt"


def load_frozen_swinunetr():
    from monai.networks.nets import SwinUNETR
    model = SwinUNETR(in_channels=1, out_channels=14, feature_size=48, use_v2=True)
    checkpoint = torch.load(str(SWINUNETR_WEIGHTS), map_location="cpu", weights_only=True)
    weight = checkpoint.get("state_dict", checkpoint)
    model_dict = model.swinViT.state_dict()
    pretrained = {}
    for k, v in weight.items():
        clean_key = k.replace("module.", "")
        if clean_key in model_dict and v.shape == model_dict[clean_key].shape:
            pretrained[clean_key] = v
    model.swinViT.load_state_dict(pretrained, strict=False)
    model.eval()
    print(f"  Loaded {len(pretrained)}/{len(model_dict)} real SwinUNETR weight tensors")
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_feature(model, volume: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    hidden_states = model.swinViT(tensor, model.normalize)
    feat = hidden_states[-1].mean(dim=[2, 3, 4])
    return feat.squeeze(0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw/lidc")
    ap.add_argument("--max_patients", type=int, default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    progress = json.loads((data_dir / "download_progress.json").read_text())
    ok_patients = [pid for pid, v in progress.items() if v.get("status") == "ok"]
    if args.max_patients:
        ok_patients = ok_patients[: args.max_patients]
    print(f"Real LIDC-IDRI patients to process: {len(ok_patients)}")

    print("Loading frozen, pretrained real SwinUNETR encoder...")
    model = load_frozen_swinunetr()

    loader = LIDCLoader(str(data_dir), target_size=(96, 96, 96))

    features, patient_ids, errors = [], [], []
    for i, pid in enumerate(ok_patients):
        try:
            volume = loader.load_ct_volume(str(data_dir / pid))
            volume = loader.preprocess_volume(volume)
            feat = extract_feature(model, volume)
            features.append(feat)
            patient_ids.append(pid)
        except Exception as e:
            errors.append((pid, str(e)))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(ok_patients)}  (errors so far: {len(errors)})")

    features = np.stack(features)
    print(f"\nReal SwinUNETR features extracted: {features.shape}  (failed: {len(errors)})")

    Path("data/raw/lidc").mkdir(parents=True, exist_ok=True)
    np.save("data/raw/lidc/swinunetr_features.npy", features)
    with open("data/raw/lidc/swinunetr_patient_ids.json", "w") as f:
        json.dump(patient_ids, f)
    if errors:
        with open("data/raw/lidc/swinunetr_errors.json", "w") as f:
            json.dump(errors, f, indent=2)
    print("Saved data/raw/lidc/swinunetr_features.npy and swinunetr_patient_ids.json")


if __name__ == "__main__":
    main()
