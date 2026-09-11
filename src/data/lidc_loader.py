"""LIDC-IDRI (Lung Image Database Consortium) data loader.

Loads CT scans from the LIDC-IDRI dataset and extracts features using
the frozen SwinUNETR encoder.

Usage:
    1. Download LIDC-IDRI from TCIA:
       https://www.cancerimagingarchive.net/collection/lidc-idri/
    
    2. Place DICOM files in data/raw/lidc/
    
    3. Run: python scripts/extract_features.py --source lidc
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    import pydicom
except ImportError:
    pydicom = None


MALIGNANCY_THRESHOLD = 3


class LIDCLoader:
    """Load and preprocess LIDC-IDRI CT scan data."""

    def __init__(self, data_dir: str, target_size: Tuple[int, ...] = (96, 96, 96)):
        self.data_dir = Path(data_dir)
        self.target_size = target_size

    def find_dicom_series(self) -> list:
        """Find all DICOM series directories."""
        series_dirs = []
        for dcm_file in self.data_dir.rglob("*.dcm"):
            series_dir = dcm_file.parent
            if series_dir not in series_dirs:
                series_dirs.append(series_dir)
        return sorted(series_dirs)

    def load_ct_volume(self, series_dir: str) -> np.ndarray:
        """Load a CT volume from a DICOM series directory."""
        if sitk is None:
            raise ImportError("SimpleITK required: pip install SimpleITK")

        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(str(series_dir))
        if not dicom_files:
            raise ValueError(f"No DICOM series found in {series_dir}")

        reader.SetFileNames(dicom_files)
        image = reader.Execute()

        image = self._resample_volume(image)

        volume = sitk.GetArrayFromImage(image).astype(np.float32)
        return volume

    def _resample_volume(self, image: "sitk.Image") -> "sitk.Image":
        """Resample volume to target size."""
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()

        new_spacing = [
            original_spacing[i] * original_size[i] / self.target_size[i]
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(self.target_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1024)

        return resampler.Execute(image)

    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Apply standard CT preprocessing (windowing + normalization)."""
        min_hu = -1350
        max_hu = 150
        volume = np.clip(volume, min_hu, max_hu)
        volume = (volume - min_hu) / (max_hu - min_hu)
        return volume

    def load_annotations(self, xml_dir: Optional[str] = None) -> dict:
        """Load LIDC nodule annotations to derive cancer labels.
        
        Malignancy ratings (1-5 scale from radiologist annotations):
        1-2: Benign
        3: Indeterminate
        4-5: Malignant
        """
        annotations = {}
        ann_dir = Path(xml_dir) if xml_dir else self.data_dir

        for xml_file in ann_dir.rglob("*.xml"):
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_file)
                root = tree.getroot()
                ns = {"lidc": "http://www.nih.gov"}

                patient_id = None
                max_malignancy = 0

                for elem in root.iter():
                    if "StudyInstanceUID" in elem.tag:
                        patient_id = elem.text
                    if "malignancy" in elem.tag.lower():
                        try:
                            mal = int(elem.text)
                            max_malignancy = max(max_malignancy, mal)
                        except (ValueError, TypeError):
                            pass

                if patient_id:
                    annotations[patient_id] = {
                        "malignancy": max_malignancy,
                        "cancer_label": 1 if max_malignancy >= MALIGNANCY_THRESHOLD else 0,
                    }
            except Exception:
                continue

        return annotations

    def extract_all_features(self, encoder, device: str = "cpu",
                              batch_size: int = 1) -> dict:
        """Extract SwinUNETR features from all CT volumes."""
        import torch

        series_dirs = self.find_dicom_series()
        print(f"Found {len(series_dirs)} CT series")

        all_features = []
        all_ids = []
        errors = []

        encoder.eval()
        encoder.to(device)

        for i, series_dir in enumerate(series_dirs):
            try:
                volume = self.load_ct_volume(str(series_dir))
                volume = self.preprocess_volume(volume)

                tensor = torch.FloatTensor(volume).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    features = encoder.extract_features(tensor)

                all_features.append(features.cpu().numpy().squeeze())
                all_ids.append(series_dir.name)

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(series_dirs)}")

            except Exception as e:
                errors.append((str(series_dir), str(e)))

        if errors:
            print(f"  Errors: {len(errors)} series failed")

        features_array = np.stack(all_features) if all_features else np.array([])

        return {
            "features": features_array,
            "series_ids": all_ids,
            "errors": errors,
        }
