# Traffic Sign Classification with Adversarial Robustness

Classification of German traffic signs (GTSRB, 43 classes) using AlexNet, with two complementary approaches to adversarial robustness: adversarial training and a full adversarial defence pipeline.

---

## Project Overview

The project is split into two main parts, each with its own notebook.

**Part 1 — Baseline & Adversarial Training** (`EvaluateAdversarialTraining.ipynb`)  
Train an AlexNet classifier on GTSRB. Then re-train it mixing clean and occlusion-attacked images at different ratios to study robustness via data augmentation alone. Evaluates across four attack types: occlusion, shadow, noise+blur, and graffiti.

**Part 2 — Adversarial Detection Pipeline** (`ComputerVision_Groupproject.ipynb`)  
A three-module defence pipeline applied on top of AlexNet:

```
Input image
    │
    ▼
AttackClassifier ──► attack probability p
    │
    ▼
LearnedDenoiser ──► corrected image
    │
    ▼
Soft Blend: p·corrected + (1−p)·original
    │
    ▼
AlexNet ──► sign class (0–42)
```

The soft blend is the key: instead of a hard if/else switch (which breaks gradients), the denoised and original images are mixed proportionally to how confident the detector is that an attack is present. This lets all three modules be fine-tuned jointly end-to-end.

Two evaluation modes:
- **Approach A** — each module loaded from its own pre-trained weights, no joint training
- **Approach B** — all modules fine-tuned end-to-end from Approach A's starting point

---

## Repository Structure

```
├── EvaluateAdversarialTraining.ipynb   Part 1: adversarial training & evaluation
├── ComputerVision_Groupproject.ipynb   Part 2: adversarial detection pipeline
│
├── architectures.py          AlexNet (43-class) + LearnedDenoiser (U-Net)
├── attack_classifier.py      AttackClassifier — binary CNN (clean vs attacked)
├── attack_detection_dataset.py  AttackDetectionDataset, DenoiserDataset, E2EDataset
├── pipeline.py               AdversarialRobustnessPipeline + training functions
├── notebook_setup.py         Path resolution + data-preparation helpers (used by Part 2)
├── evaluate.py               Evaluation helpers: accuracy, per-attack eval, comparison
├── utils.py                  TestDataset, OwnImagesDataset
├── plotting.py               Training curves + accuracy/comparison bar charts
├── requirements.txt
│
├── physical_adv_attack/
│   ├── generator.py          Attack generator (occlusion, shadow, noise_blur, graffiti)
│   ├── run.py                CLI entry point
│   ├── config.yaml           Generator settings
│   └── generated/            Generated attacked images + manifest.csv (git-ignored)
│
├── model/                    Saved model weights (*.pth git-ignored)
│   ├── first_model_weights.pth         AlexNet — clean training (Part 2 fallback)
│   ├── 100_initial_data.pth            AlexNet — clean training (Part 1)
│   ├── adv_training_0.5_occlusion.pth  AlexNet — 50% clean / 50% occlusion (Part 2)
│   ├── 50_initial_50_occlusion.pth     AlexNet — 50% clean / 50% occlusion (Part 1)
│   ├── 70_initial_30_occlusion.pth     AlexNet — 70% clean / 30% occlusion
│   ├── 30_initial_70_occlusion.pth     AlexNet — 30% clean / 70% occlusion
│   ├── 100_initial_100_occlusion.pth   AlexNet — 100% clean + 100% occlusion
│   ├── attack_classifier.pth           AttackClassifier weights
│   ├── denoiser.pth                    LearnedDenoiser weights
│   └── pipeline_e2e.pth                End-to-end pipeline (generated after e2e training)
│
├── results/                  Evaluation CSVs produced by Part 1
│   ├── <model>_test_results.csv          Per-image predictions on clean test set
│   ├── <model>_<attack>_test_results.csv Per-image predictions on attacked test set
│   ├── <model>_percentages.csv           Accuracy summary across all test sets
│   ├── <model>_own_imgs_results.csv      Predictions on own example images
│   ├── compare_adv_training.csv          Model comparison on clean test set
│   ├── compare_adv_training_initial.csv  Same/improved/worsened breakdown
│   └── compare_adv_training_occlusion.csv Model comparison on occlusion test set
│
├── data/
│   ├── gtsrb/                GTSRB dataset (downloaded automatically, git-ignored)
│   └── own_imgs/             Own traffic sign photos (12.jpeg, 13.jpeg, 33.jpeg)
│
└── plots/                    Output directory for generated plot PNGs (git-ignored)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install scikit-learn   # needed for Part 2 evaluation cells
```

### 2. Download and generate data

The GTSRB dataset downloads automatically the first time you run either notebook (via `torchvision.datasets.GTSRB`).

To generate attacked images manually:

```bash
python physical_adv_attack/run.py
```

This produces `physical_adv_attack/generated/manifest.csv` and all attacked images under `physical_adv_attack/generated/train/occlusion/`. Part 2's notebook also auto-generates these if missing.

### 3. Open a notebook

```bash
jupyter notebook EvaluateAdversarialTraining.ipynb   # Part 1
jupyter notebook ComputerVision_Groupproject.ipynb   # Part 2
```

Or open in VS Code with the Jupyter extension.

---

## Part 1 — Adversarial Training Notebook (`EvaluateAdversarialTraining.ipynb`)

Self-contained notebook covering the full training and evaluation workflow for adversarial training.

### Flags (cell 2)

```python
run_first_model          = False   # True → train baseline AlexNet from scratch
run_adversarial_training = True    # True → retrain with clean + attacked mix
running_on_colab         = False   # set True if running on Google Colab
```

Set both training flags to `False` if weights already exist in `model/`.

### What it does

| Section | Description |
|---|---|
| Dataset loading | Downloads GTSRB, builds train/val split (80/20), defines `TrainingDataset` for mixed clean+attacked batches |
| AlexNet | Defines and instantiates AlexNet (43 classes, inline definition) |
| Training | `training_loop` with Adam optimiser, saves best weights and CSV per epoch |
| `training_setup(non_adv_ratio, attack_name)` | Creates a mixed dataset at the given clean/attack ratio, runs training, saves as `<perc>_initial_<100-perc>_<attack>.pth` |
| Adversarial training runs | Trains at multiple ratios: 70/30, 50/50, 30/70, 0/100 occlusion |
| Evaluation | `test_model` evaluates on the GTSRB test set |

### Model naming convention

| Weight file | Meaning |
|---|---|
| `100_initial_data.pth` | 100% clean training data |
| `70_initial_30_occlusion.pth` | 70% clean + 30% occlusion attacked |
| `50_initial_50_occlusion.pth` | 50% clean + 50% occlusion attacked |
| `30_initial_70_occlusion.pth` | 30% clean + 70% occlusion attacked |
| `100_initial_100_occlusion.pth` | 100% clean + 100% occlusion (dataset doubled) |

---

## Part 2 — Detection Pipeline Notebook (`ComputerVision_Groupproject.ipynb`)

### Flags (cell 24)

```python
run_attack_classifier_training = False  # True → retrain from scratch
run_denoiser_training          = False  # True → retrain from scratch
run_e2e_training               = False  # True → run end-to-end fine-tuning
auto_prepare_e2e_data          = True   # auto-generate manifest/images if missing
force_regenerate_attack_data   = False  # regenerate attacked data even if manifest exists
```

**All training flags are `False` by default** — the notebook loads saved weights from `model/`. If `model/attack_classifier.pth` and `model/denoiser.pth` exist, you can run the entire notebook without training anything.

### Cell-by-cell guide

| Cells | What happens |
|---|---|
| 24 | Set flags and paths. `dataset_root` points to `data/gtsrb` and `config_path` points to the generator config. |
| 25 | Imports all modules. Uses `importlib.reload` so changes to `.py` files are picked up without restarting the kernel. |
| 28 | Defines `make_training_callback()` — a live-updating Jupyter plot that refreshes after every training epoch. |
| 29–30 | **Auto-prepare E2E assets** — checks/updates `physical_adv_attack/config.yaml`, generates attacked data if needed, resolves pretrained weight paths. |
| 31–32 | **Build datasets and loaders** — creates `AttackDetectionDataset`, `DenoiserDataset`, and `E2EDataset` with 80/20 train/val split. |
| 34 | Loads or trains the **AttackClassifier**. Shows saved training curves (BCE loss + val accuracy). |
| 36 | Loads or trains the **LearnedDenoiser**. Shows saved training curves (MSE loss). |
| 38 | Assembles **Pipeline A** — loads AlexNet (`adv_training_0.5_occlusion.pth`), AttackClassifier, and Denoiser into `AdversarialRobustnessPipeline`. |
| 44 | Assembles **Pipeline B** — deep-copies Pipeline A, then runs e2e fine-tuning if `run_e2e_training=True`, otherwise loads `model/pipeline_e2e.pth` if it exists. |
| 42 | **Evaluate AlexNet** — accuracy on clean GTSRB test set, normalised confusion matrix, per-class bar chart, training curves. |
| 44 | **Evaluate AttackClassifier** — precision, recall, F1, ROC curve, accuracy vs threshold sweep, score distribution histogram. |
| 46 | **Evaluate Denoiser** — MSE and PSNR on val set, attacked → denoised → original visual grid (8 examples). |
| 53 | **Pipeline comparison** — evaluates Baseline / Pipeline A / Pipeline B on unseen GTSRB test images at 5 attack ratios (0%, 25%, 50%, 75%, 100%). |
| 54 | Bar charts: clean vs attacked accuracy at 50% ratio, attacked accuracy vs ratio, detection accuracy vs ratio. |
| 56 | Shows individual examples where the pipeline corrects a misclassification. |

### To run end-to-end training

1. Make sure `model/attack_classifier.pth` and `model/denoiser.pth` exist.
2. Set `run_e2e_training = True` in cell 24.
3. Run cells 24 → 44. Pipeline B trains for 10 epochs and saves to `model/pipeline_e2e.pth`.

---

## Modules

### AlexNet (`architectures.py`)

Standard AlexNet adapted for 43-class GTSRB:
- 5 convolutional layers with BatchNorm
- 3 fully-connected layers (4096 → 4096 → 43)
- Input: 224×224 RGB — output: 43 logits

### AttackClassifier (`attack_classifier.py`)

Lightweight binary CNN:
- 3 conv blocks (32 → 64 → 128 channels) with BatchNorm and stride-based downsampling
- AdaptiveAvgPool → 128×4×4 → FC 256 → single logit
- Trained with BCEWithLogitsLoss (clean=0, attacked=1)
- ~900K parameters

### LearnedDenoiser (`architectures.py`)

U-Net style convolutional autoencoder, based on [arXiv 2312.03520](https://arxiv.org/abs/2312.03520):
- **Encoder**: 3→32 (stride 1), 32→64 (stride 2, 112px), 64→128 (stride 2, 56px)
- **Decoder**: ConvTranspose back to 224px with skip connections from encoder
- GELU activations throughout (avoids dying ReLU)
- **Residual output**: `x + learned_correction` — the network learns only the noise/distortion to subtract
- Trained with MSE loss on (attacked → clean) image pairs
- ~500K parameters

### AdversarialRobustnessPipeline (`pipeline.py`)

Assembles the three modules. Forward pass:
1. `attack_prob = sigmoid(AttackClassifier(x))`  — scalar per image, ∈ [0, 1]
2. `corrected = LearnedDenoiser(x)`
3. `blended = attack_prob · corrected + (1 − attack_prob) · x`  — differentiable soft switch
4. `sign_logits = AlexNet(blended)`

Returns `(sign_logits, attack_prob)`. The `.predict()` method applies `argmax` and a 0.5 threshold.

---

## Training Functions (`pipeline.py`)

All three training functions accept an optional `callback` parameter for live Jupyter plots:

```python
train_attack_classifier(classifier, train_loader, val_loader, device,
                        num_epochs=15, lr=1e-3,
                        save_path="model/attack_classifier.pth",
                        callback=None)

train_denoiser(denoiser, train_loader, val_loader, device,
               num_epochs=20, lr=1e-3,
               save_path="model/denoiser.pth",
               callback=None)

train_end_to_end(pipeline, train_loader, val_loader, device,
                 num_epochs=10, lr=1e-4, lambda_det=0.5,
                 save_path="model/pipeline_e2e.pth",
                 callback=None)
```

The e2e loss is:
```
L = CrossEntropy(sign logits, sign labels) + λ · BCE(attack logits, attack labels)
```
`lambda_det=0.5` by default.

---

## Evaluation & Plotting Scripts

These scripts are used by Part 1 and can also be run standalone.

### `evaluate.py`

| Function | Description |
|---|---|
| `eval_model_on_test_sets(weights_fname)` | Runs the model on the clean GTSRB test set and all 4 attack test sets, saves per-image CSVs to `results/` |
| `eval_attack(model, device, attack_name, weights_fname)` | Evaluates one model on one attack test set, caches result in `results/` |
| `eval_model_on_own_imgs(weights_fname)` | Runs inference on photos in `data/own_imgs/`, saves to `results/` |
| `get_accuracies_of_model(weights_fname)` | Loads accuracy summary from `results/<model>_percentages.csv` |
| `find_improved_prediction_imgs(csv_A, csv_B)` | Returns images where model A was wrong but model B was correct |
| `compare_model_predictions(initial, compare_list, out_fname)` | Produces same/improved/worsened breakdown between models |

### `utils.py`

| Class | Description |
|---|---|
| `TestDataset(attack_name=None)` | GTSRB test set (clean or attacked). Returns `(image, label, orig_fname, fname)`. |
| `OwnImagesDataset()` | Images from `data/own_imgs/`. Filename must be `<class_id>.<ext>` (e.g. `12.jpeg` → class 12). |

### `plotting.py`

| Function | Description |
|---|---|
| `plot_over_epochs(files)` | Saves accuracy-over-epochs and loss-over-epochs PNGs for a list of models |
| `plot_test_accuracies(files)` | Grouped bar chart of test accuracy across all attack types |
| `plot_comparison(...)` | Stacked bar chart showing same/improved/worsened between a baseline and comparison models |
| `plot_tricked_initial_correct_adv_trained(n_imgs)` | Saves example images where adversarial training fixed the baseline's mistakes |

Run `python plotting.py` to regenerate all comparison plots from the saved CSVs in `results/`.

---

## Dataset Generation (`physical_adv_attack/`)

The generator applies physical adversarial attacks to GTSRB images.

| Attack | Description |
|---|---|
| `occlusion` | 1–3 coloured square stickers placed randomly on the sign |
| `shadow` | Triangular shadow cast across part of the sign |
| `noise_blur` | Gaussian noise + blur inside the sign boundary |
| `graffiti` | Short text drawn on the sign surface |

Each attacked image has a corresponding entry in `manifest.csv`. To change which attacks are generated, edit `physical_adv_attack/config.yaml`:
```yaml
attacks:
  occlusion:
    enabled: true
  shadow:
    enabled: false
```

---

## Saved Weights

| File | Description |
|---|---|
| `model/first_model_weights.pth` | AlexNet — trained on clean GTSRB (Part 2 fallback name) |
| `model/100_initial_data.pth` | AlexNet — trained on clean GTSRB (Part 1 name) |
| `model/adv_training_0.5_occlusion.pth` | AlexNet — 50% clean / 50% occlusion (Part 2 name) |
| `model/50_initial_50_occlusion.pth` | AlexNet — 50% clean / 50% occlusion (Part 1 name) |
| `model/70_initial_30_occlusion.pth` | AlexNet — 70% clean / 30% occlusion |
| `model/30_initial_70_occlusion.pth` | AlexNet — 30% clean / 70% occlusion |
| `model/100_initial_100_occlusion.pth` | AlexNet — 100% clean + 100% occlusion (dataset doubled) |
| `model/attack_classifier.pth` | AttackClassifier — binary clean vs attacked (Part 2) |
| `model/denoiser.pth` | LearnedDenoiser — MSE reconstruction (Part 2) |
| `model/pipeline_e2e.pth` | Full pipeline — end-to-end fine-tuned *(generated after running e2e training)* |
