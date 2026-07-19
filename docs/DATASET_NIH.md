# NIH ChestX-ray14 — Dataset Preparation Guide (M8)

The pipeline is dataset-agnostic; only `configs/data_nih.yaml` and
`configs/model_nih.yaml` point it at NIH. The dataset is **not auto-downloaded**
(≈42 GB, licensing/hosting) — follow the steps below.

## 1. Official download

NIH ChestX-ray14 is released by the NIH Clinical Center:

- Landing page: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Also mirrored on Kaggle: https://www.kaggle.com/datasets/nih-chest-xrays/data

Download:
1. **`Data_Entry_2017.csv`** — the label/metadata file (image index, finding labels, patient ID).
2. **`images_001.tar.gz` … `images_012.tar.gz`** — the 12 image archives (112,120 PNGs).

Extract all archives so that **every PNG lives in one flat folder** (the archives
each contain an `images/` subfolder; merge them).

## 2. Expected folder structure

Relative to the repo root (or set absolute paths in `configs/data_nih.yaml`):

```
data/nih/
├── Data_Entry_2017.csv
└── images/
    ├── 00000001_000.png
    ├── 00000001_001.png
    └── ... (112,120 files, one flat directory)
```

`configs/data_nih.yaml` defaults:
```
data.nih.csv:        data/nih/Data_Entry_2017.csv
data.nih.images_dir: data/nih/images
data.nih.subset_size: null      # set an integer (patient cap) for a fast smoke run
```

## 3. Verification checklist

- [ ] `Data_Entry_2017.csv` exists and has columns `Image Index`, `Finding Labels`, `Patient ID`.
- [ ] `data/nih/images/` contains PNGs (≈112,120 for the full set).
- [ ] File names in the CSV's `Image Index` match files in `images/`.
- [ ] `pip install -r requirements.txt` succeeded (includes `pillow`, `matplotlib`).
- [ ] `python -c "from src.data.nih import load_nih_manifest; print(len(load_nih_manifest('data/nih/Data_Entry_2017.csv')))"` prints the row count.

## 4. Smoke test (fast, before the full run)

Cap the dataset to a few patients to confirm the wiring end-to-end:

1. In `configs/data_nih.yaml` set `data.nih.subset_size: 40` and `data.num_clients: 3`.
2. In `configs/eval.yaml` set `seeds: [42]` and `lambdas: [0.5]` (single quick pass).
3. Run:
   ```
   python scripts/run_federated.py    # if you point run_federated at NIH; optional
   python scripts/run_final.py        # full chain + eval + figures on the subset
   ```
   Expect `M8 FINAL OK` and `experiments/final_*.{csv,json}` + `experiments/figures/*.{png,pdf}`.
4. Restore `subset_size: null`, seeds `[42,43,44]`, and the full λ grid for the real run.

## 5. Notes

- Images are loaded as **grayscale**, resized to `data.image_size` (224) deterministically.
- Splits are **patient-level** (no patient appears in more than one split or client).
- Partitioning is **patient-level** IID or Dirichlet (`data.partition.scheme`/`alpha`).
- Labels are the 14 official findings; "No Finding" maps to an all-zero vector.
