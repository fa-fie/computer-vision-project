# Traffic Sign Classification with Adversarial Robustness

Classification of German traffic signs (GTSRB, 43 classes) using AlexNet, with a full adversarial defence pipeline combining an attack detector, a learned denoiser, and end-to-end fine-tuning.

---

## Project Overview

The project is split into two main parts:

**Part 1 — Baseline & Adversarial Training**  
Train an AlexNet classifier on GTSRB. Then re-train it mixing clean and occlusion-attacked images at different ratios to study robustness via data augmentation alone.

**Part 2 — Adversarial Detection Pipeline**  
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
ComputerVision_Groupproject.ipynb   ← main notebook (run this)

architectures.py          AlexNet (43-class sign classifier)
                          LearnedDenoiser (U-Net autoencoder)
attack_classifier.py      AttackClassifier (binary CNN: clean vs attacked)
attack_detection_dataset.py  three dataset classes:
                               AttackDetectionDataset  (binary labels)
                               DenoiserDataset         (attacked/clean pairs)
                               E2EDataset              (image + sign class + attack label)
pipeline.py               AdversarialRobustnessPipeline + training loops
evaluate.py               standalone evaluation helpers
plotting.py               loss/accuracy plot helpers
run_experiments.py        inference and visualisation scripts

physical_adv_attack/
  generator.py            attack generator (occlusion, shadow, noise_blur, graffiti)
  run.py                  CLI entry point
  config.yaml             generator settings (attack type, dataset path, etc.)

model/                    saved weights and training CSVs
  first_model_weights.pth         AlexNet — clean training
  adv_training_0.5_occlusion.pth  AlexNet — 50% adversarial mix  ← used by pipeline
  attack_classifier.pth           AttackClassifier weights
  denoiser.pth                    LearnedDenoiser weights
  pipeline_e2e.pth                End-to-end fine-tuned pipeline (generated after e2e run)

data/
  gtsrb/                  GTSRB dataset (downloaded automatically by torchvision)

physical_adv_attack/generated/
  manifest.csv            index of all generated attacked images
  train/occlusion/        attacked training images
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install scikit-learn   # needed for evaluation cells
```

### 2. Download and generate data

The GTSRB dataset downloads automatically the first time you run the notebook (via `torchvision.datasets.GTSRB`). The attacked images must be generated once:

```bash
# Edit physical_adv_attack/config.yaml first if needed (dataset_root is already set)
python physical_adv_attack/run.py
```

This produces `physical_adv_attack/generated/manifest.csv` and all attacked images under `physical_adv_attack/generated/train/occlusion/`.

### 3. Open the notebook

```bash
jupyter notebook ComputerVision_Groupproject.ipynb
```

Or open it in VS Code with the Jupyter extension.

---

## How to Use the Notebook

The notebook is divided into two main sections. Run cells top to bottom.

### Section 1–3: Baseline and Adversarial Training (cells 0–22)

These cells train AlexNet from scratch and run adversarial training experiments. Controlled by two flags in cell 2:

```python
run_first_model         = True   # train baseline AlexNet
run_adversarial_training = True  # train with occlusion-attacked mix
```

Set both to `False` if you already have the weights (`model/first_model_weights.pth`, `model/adv_training_0.5_occlusion.pth`) and just want to run the pipeline section.

### Section 4: Adversarial Detection Pipeline (cells 23–57)

Controlled by three flags in cell 24:

```python
run_attack_classifier_training = False  # True → retrain from scratch
run_denoiser_training          = False  # True → retrain from scratch
run_e2e_training               = False  # True → run end-to-end fine-tuning
```

**All three are `False` by default** — the notebook loads the saved weights from `model/`. If the weights are present, you can run the entire section without training anything.

#### Cell-by-cell guide

| Cells | What happens |
|---|---|
| 24 | Set flags and paths. `dataset_root` points to `data/gtsrb` — no change needed locally. |
| 25 | Imports all modules. Uses `importlib.reload` so changes to `.py` files are picked up without restarting the kernel. |
| 27 | Defines `make_training_callback()` — a live-updating Jupyter plot that refreshes after every training epoch. |
| 29–30 | **Dataset verification** — prints manifest stats and shows 6 clean/attacked image pairs. Run this to confirm the generator output looks correct. |
| 32 | Builds all three data loaders (80/20 train/val split, seeded). |
| 34 | Loads or trains the **AttackClassifier**. Shows saved training curves (BCE loss + val accuracy). |
| 36 | Loads or trains the **LearnedDenoiser**. Shows saved training curves (MSE loss). |
| 38 | Assembles **Pipeline A** — loads AlexNet (`adv_training_0.5_occlusion.pth`), AttackClassifier, and Denoiser; wraps them in `AdversarialRobustnessPipeline`. |
| 40 | Assembles **Pipeline B** — deep-copies Pipeline A then runs e2e fine-tuning if `run_e2e_training=True`, otherwise loads `model/pipeline_e2e.pth` if it exists. |
| 42 | **Evaluate AlexNet** — accuracy on clean GTSRB test set, normalised confusion matrix, per-class bar chart, training curves. |
| 44 | **Evaluate AttackClassifier** — precision, recall, F1, ROC curve, accuracy vs threshold sweep, score distribution histogram. |
| 46 | **Evaluate Denoiser** — MSE and PSNR on val set, attacked → denoised → original visual grid (8 examples). |
| 48 | **Pipeline comparison** — evaluates all three scenarios (Baseline / Pipeline A / Pipeline B) on the E2E val set, splits results by clean vs attacked images, produces a grouped bar chart. |
| 50 | Shows individual examples where Pipeline A corrects a misclassification that AlexNet alone gets wrong. |
| 52–55 | **Architecture diagrams** — full pipeline flow, AlexNet layer breakdown, AttackClassifier breakdown, LearnedDenoiser U-Net diagram with skip connections. |
| 57 | Parameter counts for each module and the full pipeline. |

#### To run end-to-end training

1. Make sure `model/attack_classifier.pth` and `model/denoiser.pth` exist (run those cells first or set their flags to `True`).
2. Set `run_e2e_training = True` in cell 24.
3. Run cells 24 → 40. Pipeline B will train for 10 epochs and save to `model/pipeline_e2e.pth`.

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
- GELU activations throughout (avoids dying ReLU, as recommended in the paper)
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

All three training functions accept an optional `callback` parameter:

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

The `callback` receives the full history list after each epoch. The notebook uses `make_training_callback()` to pass a live-updating matplotlib plot. All three functions save the best checkpoint automatically (best val accuracy for the classifier, best val MSE for the denoiser, best sign accuracy for the e2e pipeline).

The e2e loss is:
```
L = CrossEntropy(sign logits, sign labels) + λ · BCE(attack logits, attack labels)
```
`lambda_det=0.5` by default — reduce it if sign accuracy degrades too much.

---

## Dataset Generation (`physical_adv_attack/`)

The generator applies physical adversarial attacks to GTSRB training images. Only **occlusion** is enabled in the current `config.yaml`.

| Attack | Description |
|---|---|
| `occlusion` | 1–3 coloured square stickers placed randomly on the sign |
| `shadow` | Triangular shadow cast across part of the sign |
| `noise_blur` | Gaussian noise + blur inside the sign boundary |
| `graffiti` | Short text drawn on the sign surface |

Each attacked image has a corresponding entry in `manifest.csv` with the original path, output path, sign class, and attack parameters. The manifest is the single source of truth used by all three dataset classes.

To change which attacks are generated, edit `physical_adv_attack/config.yaml`:
```yaml
attacks:
  occlusion:
    enabled: true
  shadow:
    enabled: false   # set true to also generate shadow attacks
```

---

## Saved Weights

| File | Description |
|---|---|
| `model/first_model_weights.pth` | AlexNet — trained on clean GTSRB only |
| `model/adv_training_0.5_occlusion.pth` | AlexNet — 50% clean / 50% occlusion mix |
| `model/attack_classifier.pth` | AttackClassifier — binary clean vs attacked |
| `model/denoiser.pth` | LearnedDenoiser — MSE reconstruction |
| `model/pipeline_e2e.pth` | Full pipeline — end-to-end fine-tuned *(generated after running e2e training)* |
