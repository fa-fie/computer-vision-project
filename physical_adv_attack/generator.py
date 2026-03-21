from __future__ import annotations

import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageEnhance, ImageFilter


RESAMPLING_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")

CLASS_SHAPES = {
    0: "circle",
    1: "circle",
    2: "circle",
    3: "circle",
    4: "circle",
    5: "circle",
    6: "circle",
    7: "circle",
    8: "circle",
    9: "circle",
    10: "circle",
    11: "triangle",
    12: "diamond",
    13: "triangle_inverted",
    14: "octagon",
    15: "circle",
    16: "circle",
    17: "circle",
    18: "triangle",
    19: "triangle",
    20: "triangle",
    21: "triangle",
    22: "triangle",
    23: "triangle",
    24: "triangle",
    25: "triangle",
    26: "triangle",
    27: "triangle",
    28: "triangle",
    29: "triangle",
    30: "triangle",
    31: "triangle",
    32: "circle",
    33: "circle",
    34: "circle",
    35: "circle",
    36: "circle",
    37: "circle",
    38: "circle",
    39: "circle",
    40: "circle",
    41: "circle",
    42: "circle",
}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    split: str
    class_id: int
    width: int
    height: int
    roi_x1: int
    roi_y1: int
    roi_x2: int
    roi_y2: int
    shape: str
    sample_id: str

    @property
    def roi(self) -> tuple[int, int, int, int]:
        return (self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def as_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pick_range(rng: random.Random, values: list[float] | tuple[float, float]) -> float:
    return rng.uniform(float(values[0]), float(values[1]))


def pick_int_range(rng: random.Random, values: list[int] | tuple[int, int]) -> int:
    return rng.randint(int(values[0]), int(values[1]))


def resolve_output_root(config: dict[str, Any], config_path: Path | None) -> Path:
    output_root = Path(config.get("output_root", "./physical_adv_attack/generated")).expanduser()
    if output_root.is_absolute() or config_path is None:
        return output_root.resolve()
    return (config_path.parent.parent / output_root).resolve()


def resolve_annotation_file(dataset_root: Path, config: dict[str, Any], split: str) -> Path:
    annotation_file = str(config.get("annotation_file") or "").strip()
    if annotation_file:
        candidate = Path(annotation_file).expanduser()
        if candidate.is_absolute():
            return candidate
        return (dataset_root / candidate).resolve()

    default_name = "Test.csv" if split == "test" else "Train.csv"
    candidate = dataset_root / default_name
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(f"Could not find an annotation file for split '{split}'.")


def build_sign_mask(image_size: tuple[int, int], roi: tuple[int, int, int, int], shape: str) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = roi
    inset_x = max(1, round((x2 - x1) * 0.03))
    inset_y = max(1, round((y2 - y1) * 0.03))
    left = x1 + inset_x
    top = y1 + inset_y
    right = x2 - inset_x
    bottom = y2 - inset_y
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = max(1, right - left)
    height = max(1, bottom - top)

    if shape == "circle":
        draw.ellipse((left, top, right, bottom), fill=255)
    elif shape == "triangle":
        draw.polygon([(cx, top), (left, bottom), (right, bottom)], fill=255)
    elif shape == "triangle_inverted":
        draw.polygon([(left, top), (right, top), (cx, bottom)], fill=255)
    elif shape == "diamond":
        draw.polygon([(cx, top), (right, cy), (cx, bottom), (left, cy)], fill=255)
    elif shape == "octagon":
        radius = min(width, height) / 2
        points = []
        for index in range(8):
            angle = math.radians(22.5 + index * 45)
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        draw.polygon(points, fill=255)
    else:
        draw.rectangle((left, top, right, bottom), fill=255)

    return mask


def mask_overlap_ratio(mask: Image.Image, reference_mask: Image.Image) -> float:
    mask_pixels = list(mask.getdata())
    reference_pixels = list(reference_mask.getdata())
    reference_active = sum(1 for value in reference_pixels if value > 0)
    if reference_active == 0:
        return 0.0
    overlap = sum(
        1 for value, reference in zip(mask_pixels, reference_pixels) if value > 0 and reference > 0
    )
    return overlap / float(reference_active)


def load_samples(dataset_root: Path, config: dict[str, Any], split: str) -> list[Sample]:
    annotation_path = resolve_annotation_file(dataset_root, config, split)
    shape_map = {int(key): str(value) for key, value in (config.get("shape_map") or {}).items()}

    with annotation_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        samples: list[Sample] = []
        for index, row in enumerate(reader):
            image_path = (dataset_root / row["Path"]).resolve()
            width = int(row.get("Width") or row.get("width") or 0)
            height = int(row.get("Height") or row.get("height") or 0)
            if width <= 0 or height <= 0:
                with Image.open(image_path) as image:
                    width, height = image.size

            class_id = int(row.get("ClassId") or row.get("class_id") or 0)
            samples.append(
                Sample(
                    image_path=image_path,
                    split=split,
                    class_id=class_id,
                    width=width,
                    height=height,
                    roi_x1=int(row.get("Roi.X1") or 0),
                    roi_y1=int(row.get("Roi.Y1") or 0),
                    roi_x2=int(row.get("Roi.X2") or width),
                    roi_y2=int(row.get("Roi.Y2") or height),
                    shape=shape_map.get(class_id, CLASS_SHAPES.get(class_id, "box")),
                    sample_id=f"{split}_{index:05d}",
                )
            )

    limit = config.get("limit")
    if limit:
        return samples[: int(limit)]
    return samples


def apply_overlay(
    image: Image.Image,
    mask: Image.Image,
    color: tuple[int, int, int],
    opacity: float,
) -> Image.Image:
    overlay = Image.new("RGB", image.size, color)
    alpha = mask.point(lambda value: int(value * clamp(opacity, 0.0, 1.0)))
    return Image.composite(overlay, image, alpha)


def sample_triangle(roi: tuple[int, int, int, int], rng: random.Random) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = roi
    width = x2 - x1
    height = y2 - y1
    margin_x = width * 0.3
    margin_y = height * 0.3
    side = rng.choice(["top", "bottom", "left", "right"])

    if side == "top":
        base_y = y1 - rng.uniform(0, margin_y)
        return [
            (x1 - rng.uniform(0, margin_x), base_y),
            (x2 + rng.uniform(0, margin_x), base_y + rng.uniform(-margin_y * 0.2, margin_y * 0.2)),
            (rng.uniform(x1 - margin_x, x2 + margin_x), y2 + rng.uniform(0, margin_y)),
        ]
    if side == "bottom":
        base_y = y2 + rng.uniform(0, margin_y)
        return [
            (x1 - rng.uniform(0, margin_x), base_y),
            (x2 + rng.uniform(0, margin_x), base_y + rng.uniform(-margin_y * 0.2, margin_y * 0.2)),
            (rng.uniform(x1 - margin_x, x2 + margin_x), y1 - rng.uniform(0, margin_y)),
        ]
    if side == "left":
        base_x = x1 - rng.uniform(0, margin_x)
        return [
            (base_x, y1 - rng.uniform(0, margin_y)),
            (base_x + rng.uniform(-margin_x * 0.2, margin_x * 0.2), y2 + rng.uniform(0, margin_y)),
            (x2 + rng.uniform(0, margin_x), rng.uniform(y1 - margin_y, y2 + margin_y)),
        ]
    base_x = x2 + rng.uniform(0, margin_x)
    return [
        (base_x, y1 - rng.uniform(0, margin_y)),
        (base_x + rng.uniform(-margin_x * 0.2, margin_x * 0.2), y2 + rng.uniform(0, margin_y)),
        (x1 - rng.uniform(0, margin_x), rng.uniform(y1 - margin_y, y2 + margin_y)),
    ]


def polygon_mask(image_size: tuple[int, int], polygon: list[tuple[float, float]]) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    ImageDraw.Draw(mask).polygon(polygon, fill=255)
    return mask


def generate_shadow_attack(
    image: Image.Image,
    sample: Sample,
    rng: random.Random,
    config: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    sign_mask = build_sign_mask(image.size, sample.roi, sample.shape)
    coverage_low, coverage_high = config.get("coverage_ratio", [0.18, 0.5])
    attempts = int(config.get("attempts", 20))
    best_mask = None
    best_polygon = None
    best_fraction = 0.0

    for _ in range(attempts):
        triangle = sample_triangle(sample.roi, rng)
        candidate = ImageChops.multiply(sign_mask, polygon_mask(image.size, triangle))
        fraction = mask_overlap_ratio(candidate, sign_mask)
        if coverage_low <= fraction <= coverage_high:
            best_mask = candidate
            best_polygon = triangle
            best_fraction = fraction
            break
        if fraction > best_fraction:
            best_mask = candidate
            best_polygon = triangle
            best_fraction = fraction

    if best_mask is None or best_polygon is None:
        raise RuntimeError(f"Could not sample a shadow for {sample.image_path}.")

    blur_radius = pick_range(rng, config.get("blur_radius", [1.0, 3.0]))
    darkness = pick_range(rng, config.get("darkness_range", [0.4, 0.6]))
    softened_mask = best_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    attacked = apply_overlay(image, softened_mask, (0, 0, 0), darkness)

    return attacked, {
        "attack": "shadow",
        "darkness": round(darkness, 4),
        "blur_radius": round(blur_radius, 4),
        "mask_fraction": round(best_fraction, 4),
        "polygon": [[round(x, 2), round(y, 2)] for x, y in best_polygon],
    }


def sample_mask_point(mask: Image.Image, roi: tuple[int, int, int, int], rng: random.Random) -> tuple[int, int]:
    x1, y1, x2, y2 = roi
    pixels = mask.load()
    for _ in range(300):
        x = rng.randint(x1, max(x1, x2 - 1))
        y = rng.randint(y1, max(y1, y2 - 1))
        if pixels[x, y] > 0:
            return x, y
    return (x1 + x2) // 2, (y1 + y2) // 2


def generate_occlusion_attack(
    image: Image.Image,
    sample: Sample,
    rng: random.Random,
    config: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    sign_mask = build_sign_mask(image.size, sample.roi, sample.shape)
    sticker_count = pick_int_range(rng, config.get("sticker_count", [1, 3]))
    min_side = min(sample.roi_x2 - sample.roi_x1, sample.roi_y2 - sample.roi_y1)
    opacity = pick_range(rng, config.get("opacity_range", [0.85, 1.0]))
    color = ImageColor.getrgb(rng.choice(config.get("palette", ["#111111"])))

    sticker_mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(sticker_mask)
    sticker_boxes: list[list[int]] = []

    for _ in range(sticker_count):
        size_ratio = pick_range(rng, config.get("sticker_size_ratio", [0.12, 0.24]))
        side = max(4, int(min_side * size_ratio))
        center_x, center_y = sample_mask_point(sign_mask, sample.roi, rng)
        half = side // 2
        left = center_x - half
        top = center_y - half
        right = left + side
        bottom = top + side
        draw.rectangle((left, top, right, bottom), fill=255)
        sticker_boxes.append([left, top, right, bottom])

    clipped_mask = ImageChops.multiply(sticker_mask, sign_mask)
    attacked = apply_overlay(image, clipped_mask, color, opacity)

    return attacked, {
        "attack": "occlusion",
        "style": "square_stickers",
        "sticker_count": sticker_count,
        "sticker_boxes": sticker_boxes,
        "opacity": round(opacity, 4),
        "color_rgb": list(color),
        "mask_fraction": round(mask_overlap_ratio(clipped_mask, sign_mask), 4),
    }


def apply_physical_transform(
    image: Image.Image,
    rng: random.Random,
    config: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    if not config.get("enabled", False):
        return image, {"enabled": False}

    transformed = image
    rotation = pick_range(rng, config.get("rotation_degrees", [0.0, 0.0]))
    brightness = pick_range(rng, config.get("brightness", [1.0, 1.0]))
    contrast = pick_range(rng, config.get("contrast", [1.0, 1.0]))
    blur = pick_range(rng, config.get("gaussian_blur_radius", [0.0, 0.0]))

    if abs(rotation) > 1e-6:
        transformed = transformed.rotate(rotation, resample=RESAMPLING_BICUBIC)
    if abs(brightness - 1.0) > 1e-6:
        transformed = ImageEnhance.Brightness(transformed).enhance(brightness)
    if abs(contrast - 1.0) > 1e-6:
        transformed = ImageEnhance.Contrast(transformed).enhance(contrast)
    if blur > 0:
        transformed = transformed.filter(ImageFilter.GaussianBlur(radius=blur))

    return transformed, {
        "enabled": True,
        "rotation_degrees": round(rotation, 4),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "gaussian_blur_radius": round(blur, 4),
    }


def relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def count_total_outputs(samples: list[Sample], attacks: dict[str, Any]) -> int:
    total = 0
    for attack_name in ("shadow", "occlusion"):
        attack_config = attacks.get(attack_name, {})
        if attack_config.get("enabled", False):
            total += len(samples) * int(attack_config.get("variants_per_image", 1))
    return total


LAST_PROGRESS_PERCENT = -1


def show_progress(done: int, total: int) -> None:
    global LAST_PROGRESS_PERCENT
    if total <= 0:
        return

    percent_int = int((done / total) * 100)
    if done not in (1, total) and percent_int == LAST_PROGRESS_PERCENT:
        return

    LAST_PROGRESS_PERCENT = percent_int
    width = 30
    ratio = done / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\rProgress: [{bar}] {ratio * 100:5.1f}% ({done}/{total})")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def run_pipeline(config: dict[str, Any], config_path: Path | None = None) -> dict[str, Any]:
    dataset_root = Path(config["dataset_root"]).expanduser().resolve()
    split = str(config.get("split", "test")).lower()
    output_root = resolve_output_root(config, config_path)
    ensure_dir(output_root)
    (output_root / '.keep').write_text('', encoding='utf-8')

    samples = load_samples(dataset_root, config, split)
    attacks = config.get("attacks", {})
    seed = int(config.get("seed", 7))
    extension = str(config.get("output_extension", ".png"))
    rows: list[dict[str, Any]] = []
    total_outputs = count_total_outputs(samples, attacks)
    completed_outputs = 0

    for sample_index, sample in enumerate(samples):
        with Image.open(sample.image_path) as handle:
            image = handle.convert("RGB")

        for attack_name in ("shadow", "occlusion"):
            attack_config = attacks.get(attack_name, {})
            if not attack_config.get("enabled", False):
                continue

            variant_count = int(attack_config.get("variants_per_image", 1))
            for variant_index in range(variant_count):
                offset = 1 if attack_name == "shadow" else 2
                rng = random.Random(seed + sample_index * 1000 + variant_index * 10 + offset)
                if attack_name == "shadow":
                    attacked, attack_metadata = generate_shadow_attack(image, sample, rng, attack_config)
                else:
                    attacked, attack_metadata = generate_occlusion_attack(image, sample, rng, attack_config)

                transformed, transform_metadata = apply_physical_transform(
                    attacked,
                    rng,
                    config.get("physical_transform", {}),
                )

                output_dir = output_root / sample.split / attack_name / f"{sample.class_id:02d}"
                ensure_dir(output_dir)
                output_name = f"{sample.image_path.stem}_{attack_name}_{variant_index:02d}{extension}"
                output_path = output_dir / output_name
                transformed.save(output_path)

                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "split": sample.split,
                        "class_id": sample.class_id,
                        "shape": sample.shape,
                        "attack": attack_name,
                        "variant_index": variant_index,
                        "original_path": relative_to(sample.image_path, dataset_root),
                        "output_path": relative_to(output_path, output_root),
                        "roi_x1": sample.roi_x1,
                        "roi_y1": sample.roi_y1,
                        "roi_x2": sample.roi_x2,
                        "roi_y2": sample.roi_y2,
                        "attack_parameters": as_json(attack_metadata),
                        "transform_parameters": as_json(transform_metadata),
                    }
                )

                completed_outputs += 1
                show_progress(completed_outputs, total_outputs)

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "sample_id",
            "split",
            "class_id",
            "shape",
            "attack",
            "variant_index",
            "original_path",
            "output_path",
            "roi_x1",
            "roi_y1",
            "roi_x2",
            "roi_y2",
            "attack_parameters",
            "transform_parameters",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "split": split,
        "num_input_samples": len(samples),
        "num_generated_samples": len(rows),
        "manifest_path": str(manifest_path),
    }
