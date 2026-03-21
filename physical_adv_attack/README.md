# Physical Adversarial Attack

Small generator for traffic sign attack images.

It creates two physical-style attacks:
- `shadow`: a dark triangular shadow placed over part of the sign
- `occlusion`: one or more square sticker patches placed at random positions on the sign

## Files

- `run.py`: small command line entry point
- `generator.py`: dataset loading, attack generation, saving images, and manifest writing
- `config.yaml`: settings for dataset path, split, output folder, and attack parameters

## How to run

From the root of `computer-vision-project`:

```bash
python3 physical_adv_attack/run.py --config physical_adv_attack/config.yaml
```

Generated images are written to `physical_adv_attack/generated/`.
A `manifest.csv` file is also written there with one row per generated image.

## How to change `config.yaml`

Main fields:

- `dataset_root`: path to the dataset root folder
- `annotation_file`: leave empty for standard GTSRB-style `Train.csv` and `Test.csv`
- `output_root`: where generated images and `manifest.csv` are saved
- `split`: use `test` or `train`
- `limit`: optional number of input images to process. Leave empty to process all images

Example for a local dataset:

```yaml
dataset_root: /path/to/dataset
split: test
```

Example for training data:

```yaml
dataset_root: /path/to/dataset
split: train
```

## How to control the attacks

### Shadow attack

Inside `config.yaml`:

- `darkness_range`: how dark the shadow is
- `coverage_ratio`: how much of the sign can be covered by the shadow
- `blur_radius`: how soft the shadow border is
- `variants_per_image`: number of shadow versions to create per image

### Occlusion attack

Inside `config.yaml`:

- `sticker_count`: number of square stickers to place on the sign
- `sticker_size_ratio`: sticker size relative to the sign size
- `opacity_range`: how opaque the stickers are
- `palette`: possible sticker colors
- `variants_per_image`: number of occlusion versions to create per image

## What the code does

The code reads the traffic sign images and ROI coordinates from the dataset CSV.
Then it builds a mask for the sign shape, creates an attack only inside that sign area,
and optionally applies small physical transforms such as rotation, brightness change,
contrast change, and slight blur.

### Shadow implementation summary

The shadow attack samples a triangle around the sign area, keeps only the part that
falls on the sign, blurs the mask, and darkens the covered region.

### Occlusion implementation summary

The occlusion attack samples one or more square patches, places them at random valid
positions on the sign, clips them to the sign mask, and overlays them as sticker-like
blocks.

## Literature background

This implementation is a simple practical generator inspired by the physical traffic-sign
attack literature. It is not a full reproduction of any paper.

The design is mainly based on these references:

- Yiqi Zhong, Xianming Liu, Deming Zhai, Junjun Jiang, Xiangyang Ji, "Shadows can be Dangerous: Stealthy and Effective Physical-world Adversarial Attack by Natural Phenomenon", 2022.
- Svetlana Pavlitska, Nico Lambing, J. Marius Zollner, "Adversarial Attacks on Traffic Sign Recognition: A Survey", 2023.
- Haojie Jia, Te Hu, Haowen Li, Long Jin, Chongshi Xin, Yuchi Yao, Jiarui Xiao, "The Outline of Deception: Physical Adversarial Attacks on Traffic Signs Using Edge Patches", 2025.
