# Adversarial Detection E2e Pipeline

## Idea

Instead of only training the sign classifier on attacked images, we add two stages before it:

1. **AttackClassifier** — small binary CNN that says whether the image was attacked
2. **GaussianDenoiser** — tries to clean up the damage
3. **AlexNet** — classifies the sign as before

The tricky part is keeping everything differentiable so we can train all three together. We do this with a soft blend:

```
output = p * denoised_image + (1 - p) * original_image
```

where `p` is the attack probability. This way gradients can flow back through the whole pipeline at once.

## Training plan

**Phase 1** — train `AttackClassifier` alone
Dataset: `manifest.csv` from the generator, clean images get label 0, attacked get label 1
Loss: `BCEWithLogitsLoss`
Notebook: set `run_attack_classifier_training = True` in Section 4

**Phase 2** — implement `GaussianDenoiser` (teammate's task)
Needs to be an `nn.Module` that takes and returns `(B, 3, 224, 224)` tensors.
Plug it in at the `gaussian_denoiser` argument of `AdversarialRobustnessPipeline`.

**Phase 3** — end-to-end fine-tuning
Dataset: `E2EDataset` — same manifest, returns `(image, sign_class, attack_label)`
Loss: `CrossEntropy(sign) + 0.5 * BCE(attack detection)`
Notebook: set `run_e2e_training = True` in Section 4 (needs Phase 2 first)

## New files

- `attack_classifier.py` — the binary CNN
- `attack_detection_dataset.py` — `AttackDetectionDataset` and `E2EDataset`
- `pipeline.py` — pipeline class + `train_attack_classifier` + `train_end_to_end`

## Data

Everything is built from `physical_adv_attack/generated/manifest.csv`.
The manifest has one row per generated image with `original_path` (clean) and `output_path` (attacked), so no extra work needed to get the labels.

Run the generator if you haven't already:
```bash
python physical_adv_attack/run.py
```
