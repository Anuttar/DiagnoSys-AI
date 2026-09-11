# DiagnoSys-AI: Multimodal Disease Prediction and Clinical Decision Support

A research project validating multimodal cancer diagnosis on **real, publicly available clinical data**: LIDC-IDRI (CT imaging) for cancer detection, and TCGA-LUAD/LUSC (pathology reports + structured clinical fields) for cancer stage and histological subtype classification.

A fully unified three-modality architecture (CT + pathology text + lifestyle tabular data, fused via cross-modal attention) was designed and demonstrated on a **synthetic proof-of-concept dataset** only, since no public dataset links all three modalities to the same real patients. Real-world validation is therefore conducted as two independent, real-data tracks described below.

## Real-Data Validation Pipeline

```
Track A: Imaging (LIDC-IDRI, n=1,010 patients, ~119.3 GB DICOM)
  Real CT volumes (TCIA/NBIA) --> lung windowing (-1350 to 150 HU)
    --> Frozen SwinUNETR (whole-volume, 768-d)         --\
    --> Frozen oohtmeel Swin-Tiny (5-slice, max-pooled)  --> average-probability ensemble
                                                              --> Cancer detection (65.0% acc, AUROC 0.648)

Track B: Text + Tabular (TCGA-LUAD/LUSC, n=694-775 patients)
  Real pathology report PDFs (GDC) --> pdfplumber text extraction
    --> Frozen BiomedBERT (CLS + mean-pooled, 768-d x2) --> PCA
    --> Regex/keyword clinical features (10-d)
  Real structured clinical fields --> one-hot + missing-indicator tabular features
    --> concatenated fusion vector
    --> LogisticRegression / HistGradientBoosting / SoftVoteEnsemble
        --> Stage classification (49.71% acc, macro-F1 0.284)
        --> Subtype classification (73.42% acc, macro-F1 0.734)
```

Both tracks keep all pretrained encoders (SwinUNETR, BiomedBERT, oohtmeel) entirely **frozen** - no fine-tuning - and fit only lightweight classical classifiers on top, so the real-data tracks require no GPU for classifier training/evaluation.

## Key Results

| Task | Configuration | Accuracy | Macro-F1 | AUROC | n |
|---|---|---|---|---|---|
| Cancer detection (clean-label) | SwinUNETR + oohtmeel ensemble | 65.0% | 0.576 | 0.648 | 769 |
| Cancer detection (all cases) | SwinUNETR + oohtmeel ensemble | 63.2% | 0.535 | 0.608 | 1,010 |
| Stage classification (I-IV) | fusion_pca_regex + HistGradientBoosting | 49.71% | 0.284 | – | 694 |
| Histological subtype (LUAD vs LUSC) | fusion_pca_regex + SoftVoteEnsemble | 73.42% | 0.734 | – | 775 |

Notable honest findings: a learned stacking meta-learner **underperformed** simple probability averaging for the imaging ensemble (AUROC 0.596 vs. 0.648); a T/N/M-inclusive staging model reaches 97.27% accuracy but is flagged as circular (AJCC stage is deterministically derived from T/N/M) and excluded from headline results. See `results/*.json` for full details.

## Installation

```bash
cd DiagnoSys-AI
pip install -r requirements.txt
```

## Usage

### 1. Download real data

```bash
# LIDC-IDRI CT volumes + radiologist annotations (TCIA/NBIA, ~119.3 GB)
python scripts/download_lidc_full.py --output_dir data/raw/lidc

# TCGA-LUAD/LUSC clinical records + pathology report PDFs (GDC)
python scripts/download_tcga.py --include_reports
```

### 2. Imaging track (LIDC-IDRI)

```bash
# Run the pretrained oohtmeel lung-cancer CT classifier on real DICOM volumes
python scripts/evaluate_real_lidc.py --data_dir data/raw/lidc

# Extract frozen SwinUNETR whole-volume features (needed for the ensemble)
python scripts/extract_real_swinunetr_volumes.py --data_dir data/raw/lidc

# Combine both real pretrained models into an ensemble
python scripts/ensemble_real_lidc.py
```

### 3. Text-and-tabular track (TCGA-LUAD/LUSC)

```bash
# Fit classical classifiers on frozen BiomedBERT embeddings + tabular + regex features
python scripts/train_real_tcga.py
```

## Datasets

| Source | Modality | Real Size | Usage |
|--------|----------|-----------|-------|
| LIDC-IDRI (TCIA/NBIA) | 3D CT DICOM volumes | 1,010 patients (~119.3 GB) | Binary cancer detection |
| TCGA-LUAD/LUSC (GDC) | Pathology report PDFs + clinical fields | 1,089 clinical records / 1,003 PDFs | Stage (n=694) and subtype (n=775) classification |

No synthetic data is used in any script in this folder. The archived synthetic proof-of-concept pipeline (data generation, full cross-modal attention architecture, training/evaluation scripts) lives outside this project directory and is referenced only in the accompanying thesis document as a design-only, non-real-data demonstration.

## Pretrained Models Used (All Frozen)

| Model | Role |
|---|---|
| SwinUNETR (feature_size=48, MONAI) | Whole-volume CT feature extraction |
| oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans (HuggingFace) | Independent 2D slice malignancy probability |
| BiomedBERT (HuggingFace, PubMed-pretrained) | Pathology report text encoding |

## Limitations

- No real dataset links CT imaging, pathology reports, and lifestyle data to the same patients, so the fully unified three-modality architecture is validated only on synthetic data, not here.
- No explainability (attention maps, Grad-CAM, SHAP) is implemented for the real-data tracks; the classifiers used are simple linear/tree-based models on frozen embeddings.
- Real-data results are modest but genuine and non-circular; see `results/` and the accompanying thesis for full discussion, error analysis, and ablations.

## Citation

```
Jain, A. (2026). DiagnoSys-AI: Multimodal Disease Prediction and Clinical
Decision Support. MSc Thesis, LJMU.
```
