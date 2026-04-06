# Physical Adversarial Attack Pipeline

Generates perturbed traffic-sign images for iterative adversarial robustness research.
Each attack is self-contained and the pipeline is designed to make adding new attacks
as frictionless as possible.

---

## Quick start

```bash
python physical_adv_attack/run.py --config physical_adv_attack/config.yaml
```

Output images and a `manifest.csv` are written to `output_root` (configured in YAML).

---

## Project structure

```
physical_adv_attack/
├── generator.py      # Pipeline + all attack implementations
├── run.py            # CLI entry point
└── config.yaml       # All runtime parameters
```

If you are using the *Kaggle* version of the dataset:
```
data/
├── Test.csv          # Annotation file (Path, ClassId, Roi.*, Width, Height)
├── Train.csv
└── ...               # Image folders referenced by the CSV
```

If you are using the *Pytorch* version of the dataset:
```
data/
├── GT-final_test.csv
├── gtsrb/
├──── GTSRB/
├────── Final_Test
├────── Training
├────── ...
└── ...
```

---

## How attacks work

Every attack is a subclass of `BaseAttack` with a single method to implement:

```python
class BaseAttack(ABC):
    name: str   # matches the key used in config.yaml

    @abstractmethod
    def apply(
        self,
        image_path: Path,
        sample: Sample,
        rng: random.Random,
        config: dict[str, Any],
    ) -> tuple[Image.Image, dict[str, Any]]:
        ...
```

The returned dict is serialised to JSON and stored in `manifest.csv` under
`attack_parameters`, giving you full reproducibility for every generated image.

---

## Adding a new attack

**Step 1 — Write the class** in `generator.py`:

```python
class MyAttack(BaseAttack):
    """One-line description.

    Config keys
    -----------
    my_param : [0.1, 0.9]  — what it does
    """
    name = "my_attack"

    def apply(self, image_path, sample, rng, config):
        image = _load_rgb(image_path)
        sign_mask = build_sign_mask(image.size, sample.roi, sample.shape)

        # ... your logic here, using rng for all random decisions ...

        return attacked_image, {"attack": self.name, "my_param": value}
```

**Step 2 — Register it** (one line at the bottom of `generator.py`):

```python
ATTACK_REGISTRY: dict[str, BaseAttack] = {
    attack.name: attack
    for attack in [
        ShadowAttack(),
        OcclusionAttack(),
        NoiseBlurAttack(),
        GraffitiAttack(),
        MyAttack(),       # <-- add here
    ]
}
```

**Step 3 — Enable it in `config.yaml`**:

```yaml
attacks:
  my_attack:
    enabled: true
    variants_per_image: 3
    my_param: [0.1, 0.9]
```

Nothing else changes. The pipeline loop discovers attacks from the registry automatically.

---

## Built-in attacks

### `shadow`
Casts a triangular shadow across part of the sign by overlaying a blurred,
semi-transparent black polygon.

| Key | Default | Description |
|---|---|---|
| `coverage_ratio` | `[0.18, 0.5]` | Target fraction of sign pixels to cover |
| `attempts` | `20` | Max triangles sampled before falling back to best found |
| `darkness_range` | `[0.4, 0.6]` | Shadow opacity |
| `blur_radius` | `[1.0, 3.0]` | Gaussian blur on shadow edge |

---

### `occlusion`
Places solid-colour square stickers on the sign, clipped to the sign boundary.

| Key | Default | Description |
|---|---|---|
| `sticker_count` | `[1, 3]` | Number of stickers |
| `sticker_size_ratio` | `[0.12, 0.24]` | Sticker side as fraction of min ROI dimension |
| `opacity_range` | `[0.85, 1.0]` | Sticker opacity |
| `palette` | `["#111111"]` | Hex colour pool |

---

### `noise_blur`
Adds Gaussian pixel noise and/or blur within the sign region.

| Key | Default | Description |
|---|---|---|
| `noise_std` | `[10, 40]` | Noise standard deviation (0–255 scale) |
| `blur_radius` | `[0.0, 2.0]` | Gaussian blur radius |

---

### `graffiti`
Renders a short text string on the sign at a random position.

| Key | Default | Description |
|---|---|---|
| `texts` | `["X", "?", "//"]` | Pool of strings to sample from |
| `font_size_ratio` | `[0.2, 0.5]` | Font size as fraction of min ROI dimension |
| `opacity_range` | `[0.6, 0.95]` | Text opacity |
| `palette` | `["#ffffff", "#000000"]` | Text colour pool |
| `font_path` | `null` | Absolute path to a `.ttf` file; uses PIL default if null |

---

## Config reference

```yaml
# --- Dataset ---
dataset_root: /path/to/data          # Root folder; image paths in the CSV are relative to this
dataset_type: pytorch                # "kaggle" or "pytorch"; the datasets have different file structures
annotation_file: Test.csv            # Only used for Kaggle dataset (else leave empty): Relative to dataset_root, or absolute
split: test                          # "test" or "train"
limit:                               # Optional: process only the first N samples

# --- Output ---
output_root: ./physical_adv_attack/generated
output_extension: .png
seed: 7                              # Global seed; each (sample, attack, variant) gets a unique derived seed

# --- Per-sign shape overrides (optional) ---
shape_map: {}                        # e.g. {99: "circle"} overrides CLASS_SHAPES for class 99

# --- Post-attack physical augmentation ---
physical_transform:
  enabled: true
  rotation_degrees: [-5.0, 5.0]
  brightness: [0.92, 1.08]
  contrast: [0.92, 1.08]
  gaussian_blur_radius: [0.0, 1.0]

# --- Attacks ---
attacks:
  shadow:
    enabled: true
    variants_per_image: 1
    darkness_range: [0.4, 0.6]
    coverage_ratio: [0.18, 0.5]
    blur_radius: [1.0, 3.0]
    attempts: 20

  occlusion:
    enabled: true
    variants_per_image: 1
    sticker_count: [1, 3]
    sticker_size_ratio: [0.12, 0.24]
    opacity_range: [0.85, 1.0]
    palette: ["#111111", "#cc0000", "#ffffff"]

  noise_blur:
    enabled: false
    variants_per_image: 1
    noise_std: [10, 40]
    blur_radius: [0.0, 2.0]

  graffiti:
    enabled: false
    variants_per_image: 1
    texts: ["X", "?", "//", "NO"]
    font_size_ratio: [0.2, 0.5]
    opacity_range: [0.6, 0.95]
    palette: ["#ffffff", "#000000", "#ff0000"]
    font_path: null
```

---

## Manifest output

`manifest.csv` has one row per generated image:

| Column | Description |
|---|---|
| `sample_id` | `{split}_{index:05d}` |
| `split` | `test` / `train` |
| `class_id` | Traffic sign class |
| `shape` | Inferred sign shape |
| `attack` | Attack name |
| `variant_index` | 0-indexed variant number |
| `original_path` | Source image path (relative to `dataset_root`) |
| `output_path` | Output image path (relative to `output_root`) |
| `roi_*` | Bounding box coordinates |
| `attack_parameters` | JSON blob of attack-specific parameters |
| `transform_parameters` | JSON blob of physical transform parameters |

---

## Reproducibility

Each `(sample_index, attack_name, variant_index)` triple deterministically derives
its own RNG seed from the global `seed`:

```python
rng = random.Random(seed + sample_index * 1000 + variant_index * 10 + attack_offset)
```

Re-running with the same config and seed always produces identical output.