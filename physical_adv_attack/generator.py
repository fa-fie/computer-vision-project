"""
generator.py
============
Physical adversarial attack pipeline for traffic-sign images.

Architecture overview
---------------------
Every attack is a subclass of ``BaseAttack``.  To add a new attack:

1. Subclass ``BaseAttack`` and implement ``apply()``.
2. Register it in ``ATTACK_REGISTRY`` at the bottom of this file.
3. Add a matching key under ``attacks:`` in your YAML config.

That's it — the pipeline loop in ``run_pipeline()`` picks it up automatically.

Attack contract
---------------
``apply(image_path, sample, rng, config)  ->  (Image, metadata_dict)``

*  Receives the path to the original image (not a pre-loaded PIL object) so
   each attack controls its own I/O.
*  Returns the attacked PIL image **and** a plain dict of reproducibility
   metadata that gets written to manifest.csv.
*  Must not mutate ``sample`` or ``config``.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import (
    Image,
    ImageChops,
    ImageColor,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageFont,
)

RESAMPLING_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")

# ---------------------------------------------------------------------------
# Sign-shape lookup
# ---------------------------------------------------------------------------
CLASS_SHAPES: dict[int, str] = {
    **{i: "circle" for i in (*range(11), *range(15, 18), *range(32, 43))},
    11: "triangle",
    12: "diamond",
    13: "triangle_inverted",
    14: "octagon",
    **{i: "triangle" for i in range(18, 32)},
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    """Represents one annotated image from the dataset CSV."""

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


# ---------------------------------------------------------------------------
# Abstract base class — every attack inherits from this
# ---------------------------------------------------------------------------

class BaseAttack(ABC):
    """
    Base class for all physical adversarial attacks.

    Subclasses only need to implement ``apply()``.  Everything else
    (registration, pipeline integration) is handled externally.
    """

    #: Human-readable name used as the key in config YAML and manifest CSV.
    name: str

    @abstractmethod
    def apply(
        self,
        image_path: Path,
        sample: Sample,
        rng: random.Random,
        config: dict[str, Any],
    ) -> tuple[Image.Image, dict[str, Any]]:
        """
        Apply the attack to a single image.

        Parameters
        ----------
        image_path:
            Absolute path to the source image file.
        sample:
            Metadata for this image (ROI, class, shape, ...).
        rng:
            Seeded RNG — use this for all random decisions so results are
            reproducible given the same seed.
        config:
            The attack-specific sub-dict from the YAML config
            (e.g. ``attacks.shadow``).

        Returns
        -------
        attacked_image:
            Modified PIL image in RGB mode.
        metadata:
            Flat dict of attack parameters written to manifest.csv.
            Always include ``"attack": self.name``.
        """


# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------

def _load_rgb(image_path: Path) -> Image.Image:
    """Open an image file and return it as an RGB PIL image."""
    with Image.open(image_path) as handle:
        return handle.convert("RGB")


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to [``low``, ``high``]."""
    return max(low, min(high, value))


def pick_range(rng: random.Random, values: list[float] | tuple[float, float]) -> float:
    """Sample a float uniformly from a [min, max] pair."""
    return rng.uniform(float(values[0]), float(values[1]))


def pick_int_range(rng: random.Random, values: list[int] | tuple[int, int]) -> int:
    """Sample an int uniformly from a [min, max] pair (inclusive)."""
    return rng.randint(int(values[0]), int(values[1]))


def apply_overlay(
    image: Image.Image,
    mask: Image.Image,
    color: tuple[int, int, int],
    opacity: float,
) -> Image.Image:
    """
    Composite a solid-colour layer onto ``image`` using ``mask`` as alpha.

    Parameters
    ----------
    mask:
        Greyscale mask (L mode). 255 = fully covered, 0 = original pixel.
    opacity:
        Global opacity multiplier applied on top of the mask values.
    """
    overlay = Image.new("RGB", image.size, color)
    alpha = mask.point(lambda v: int(v * clamp(opacity, 0.0, 1.0)))
    return Image.composite(overlay, image, alpha)


def build_sign_mask(
    image_size: tuple[int, int],
    roi: tuple[int, int, int, int],
    shape: str,
) -> Image.Image:
    """
    Return a greyscale mask (255 inside the sign, 0 outside) whose shape
    matches the traffic-sign type (circle, triangle, octagon, ...).

    A small inset (3 % of ROI dimensions) is applied to avoid painting
    right on the sign border.
    """
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = roi

    # 3 % inset to stay within the sign boundary
    inset_x = max(1, round((x2 - x1) * 0.03))
    inset_y = max(1, round((y2 - y1) * 0.03))
    left, top, right, bottom = x1 + inset_x, y1 + inset_y, x2 - inset_x, y2 - inset_y
    cx, cy = (left + right) / 2, (top + bottom) / 2
    width, height = max(1, right - left), max(1, bottom - top)

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
        points = [
            (cx + radius * math.cos(math.radians(22.5 + i * 45)),
             cy + radius * math.sin(math.radians(22.5 + i * 45)))
            for i in range(8)
        ]
        draw.polygon(points, fill=255)
    else:
        draw.rectangle((left, top, right, bottom), fill=255)

    return mask


def mask_overlap_ratio(mask: Image.Image, reference_mask: Image.Image) -> float:
    """
    Fraction of ``reference_mask``'s active pixels that are also active in
    ``mask``.  Used to measure how much of the sign is covered by an attack.
    """
    ref_pixels = list(reference_mask.getdata())
    ref_active = sum(1 for v in ref_pixels if v > 0)
    if ref_active == 0:
        return 0.0
    overlap = sum(
        1 for m, r in zip(mask.getdata(), ref_pixels) if m > 0 and r > 0
    )
    return overlap / float(ref_active)


def polygon_mask(
    image_size: tuple[int, int],
    polygon: list[tuple[float, float]],
) -> Image.Image:
    """Return a filled greyscale mask for an arbitrary polygon."""
    mask = Image.new("L", image_size, 0)
    ImageDraw.Draw(mask).polygon(polygon, fill=255)
    return mask


def sample_mask_point(
    mask: Image.Image,
    roi: tuple[int, int, int, int],
    rng: random.Random,
) -> tuple[int, int]:
    """
    Sample a random pixel that is active (> 0) inside the masked ROI.
    Falls back to the ROI centre after 300 failed attempts.
    """
    x1, y1, x2, y2 = roi
    pixels = mask.load()
    for _ in range(300):
        x = rng.randint(x1, max(x1, x2 - 1))
        y = rng.randint(y1, max(y1, y2 - 1))
        if pixels[x, y] > 0:
            return x, y
    return (x1 + x2) // 2, (y1 + y2) // 2


# ---------------------------------------------------------------------------
# Attack 1 — Shadow
# ---------------------------------------------------------------------------

class ShadowAttack(BaseAttack):
    """
    Simulates a cast shadow across part of the traffic sign.

    A random triangle is sampled so that it overlaps the sign by a
    configurable fraction (``coverage_ratio``).  The overlapping region is
    darkened with a Gaussian-blurred, semi-transparent black overlay.

    Config keys (all optional, defaults shown)
    ------------------------------------------
    coverage_ratio  : [0.18, 0.5]   — desired [min, max] sign coverage
    attempts        : 20            — how many triangles to try before
                                      giving up and using the best found
    darkness_range  : [0.4, 0.6]    — opacity of the shadow overlay
    blur_radius     : [1.0, 3.0]    — Gaussian blur on the shadow edge
    """

    name = "shadow"

    def apply(
        self,
        image_path: Path,
        sample: Sample,
        rng: random.Random,
        config: dict[str, Any],
    ) -> tuple[Image.Image, dict[str, Any]]:
        image = _load_rgb(image_path)
        sign_mask = build_sign_mask(image.size, sample.roi, sample.shape)

        coverage_low, coverage_high = config.get("coverage_ratio", [0.18, 0.5])
        attempts = int(config.get("attempts", 20))

        best_mask, best_polygon, best_fraction = None, None, 0.0

        for _ in range(attempts):
            triangle = _sample_triangle(sample.roi, rng)
            candidate = ImageChops.multiply(sign_mask, polygon_mask(image.size, triangle))
            fraction = mask_overlap_ratio(candidate, sign_mask)

            # Accept immediately if within the desired coverage band
            if coverage_low <= fraction <= coverage_high:
                best_mask, best_polygon, best_fraction = candidate, triangle, fraction
                break

            # Otherwise keep the best-so-far
            if fraction > best_fraction:
                best_mask, best_polygon, best_fraction = candidate, triangle, fraction

        if best_mask is None or best_polygon is None:
            raise RuntimeError(f"Could not sample a shadow for {image_path}.")

        blur_radius = pick_range(rng, config.get("blur_radius", [1.0, 3.0]))
        darkness = pick_range(rng, config.get("darkness_range", [0.4, 0.6]))
        softened = best_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        attacked = apply_overlay(image, softened, (0, 0, 0), darkness)

        return attacked, {
            "attack": self.name,
            "darkness": round(darkness, 4),
            "blur_radius": round(blur_radius, 4),
            "mask_fraction": round(best_fraction, 4),
            "polygon": [[round(x, 2), round(y, 2)] for x, y in best_polygon],
        }


def _sample_triangle(
    roi: tuple[int, int, int, int],
    rng: random.Random,
) -> list[tuple[float, float]]:
    """
    Sample a triangle that enters the ROI from a random side.

    One vertex is anchored outside the ROI on the chosen side; the other two
    spread across the opposite side.  A 30 % margin is added around the ROI
    so the triangle can extend slightly beyond it.
    """
    x1, y1, x2, y2 = roi
    mx = (x2 - x1) * 0.3   # horizontal margin
    my = (y2 - y1) * 0.3   # vertical margin
    side = rng.choice(["top", "bottom", "left", "right"])

    if side == "top":
        by = y1 - rng.uniform(0, my)
        return [
            (x1 - rng.uniform(0, mx), by),
            (x2 + rng.uniform(0, mx), by + rng.uniform(-my * 0.2, my * 0.2)),
            (rng.uniform(x1 - mx, x2 + mx), y2 + rng.uniform(0, my)),
        ]
    if side == "bottom":
        by = y2 + rng.uniform(0, my)
        return [
            (x1 - rng.uniform(0, mx), by),
            (x2 + rng.uniform(0, mx), by + rng.uniform(-my * 0.2, my * 0.2)),
            (rng.uniform(x1 - mx, x2 + mx), y1 - rng.uniform(0, my)),
        ]
    if side == "left":
        bx = x1 - rng.uniform(0, mx)
        return [
            (bx, y1 - rng.uniform(0, my)),
            (bx + rng.uniform(-mx * 0.2, mx * 0.2), y2 + rng.uniform(0, my)),
            (x2 + rng.uniform(0, mx), rng.uniform(y1 - my, y2 + my)),
        ]
    bx = x2 + rng.uniform(0, mx)
    return [
        (bx, y1 - rng.uniform(0, my)),
        (bx + rng.uniform(-mx * 0.2, mx * 0.2), y2 + rng.uniform(0, my)),
        (x1 - rng.uniform(0, mx), rng.uniform(y1 - my, y2 + my)),
    ]


# ---------------------------------------------------------------------------
# Attack 2 — Occlusion (square stickers)
# ---------------------------------------------------------------------------

class OcclusionAttack(BaseAttack):
    """
    Places one or more solid-coloured square stickers on the sign.

    Each sticker is centred on a randomly sampled active pixel of the sign
    mask, clipped so it never extends outside the sign boundary.

    Config keys (all optional, defaults shown)
    ------------------------------------------
    sticker_count      : [1, 3]         — number of stickers per image
    sticker_size_ratio : [0.12, 0.24]   — sticker side as fraction of the
                                          shortest ROI dimension
    opacity_range      : [0.85, 1.0]    — sticker opacity
    palette            : ["#111111"]    — list of hex colours to pick from
    """

    name = "occlusion"

    def apply(
        self,
        image_path: Path,
        sample: Sample,
        rng: random.Random,
        config: dict[str, Any],
    ) -> tuple[Image.Image, dict[str, Any]]:
        image = _load_rgb(image_path)
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
            cx, cy = sample_mask_point(sign_mask, sample.roi, rng)
            half = side // 2
            box = [cx - half, cy - half, cx - half + side, cy - half + side]
            draw.rectangle(box, fill=255)
            sticker_boxes.append(box)

        # Clip stickers to sign area only
        clipped = ImageChops.multiply(sticker_mask, sign_mask)
        attacked = apply_overlay(image, clipped, color, opacity)

        return attacked, {
            "attack": self.name,
            "style": "square_stickers",
            "sticker_count": sticker_count,
            "sticker_boxes": sticker_boxes,
            "opacity": round(opacity, 4),
            "color_rgb": list(color),
            "mask_fraction": round(mask_overlap_ratio(clipped, sign_mask), 4),
        }


# ---------------------------------------------------------------------------
# Attack 3 — Noise / blur
# ---------------------------------------------------------------------------

class NoiseBlurAttack(BaseAttack):
    """
    Adds Gaussian noise and/or blur to the sign region.

    Both effects are applied within the sign mask only, leaving the
    background untouched.

    Config keys (all optional, defaults shown)
    ------------------------------------------
    noise_std       : [10, 40]      — standard deviation of Gaussian noise
                                      (0-255 scale); set both ends to 0 to
                                      disable noise entirely
    blur_radius     : [0.0, 2.0]    — Gaussian blur radius applied after
                                      noise; set to [0, 0] to skip blur
    """

    name = "noise_blur"

    def apply(
        self,
        image_path: Path,
        sample: Sample,
        rng: random.Random,
        config: dict[str, Any],
    ) -> tuple[Image.Image, dict[str, Any]]:
        image = _load_rgb(image_path)
        sign_mask = build_sign_mask(image.size, sample.roi, sample.shape)

        noise_std = pick_range(rng, config.get("noise_std", [10, 40]))
        blur_radius = pick_range(rng, config.get("blur_radius", [0.0, 2.0]))

        # --- Gaussian noise ---
        # Generate per-pixel noise offsets and blend onto original
        noisy = image.copy()
        if noise_std > 0:
            pixels = list(image.getdata())  # list of (R, G, B) tuples
            noisy_pixels = [
                tuple(
                    int(clamp(channel + rng.gauss(0, noise_std), 0, 255))
                    for channel in pixel
                )
                for pixel in pixels
            ]
            noisy.putdata(noisy_pixels)

        # --- Blur ---
        blurred = (
            noisy.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            if blur_radius > 0
            else noisy
        )

        # --- Composite: apply effect inside sign mask only ---
        attacked = Image.composite(blurred, image, sign_mask)

        return attacked, {
            "attack": self.name,
            "noise_std": round(noise_std, 4),
            "blur_radius": round(blur_radius, 4),
            "mask_fraction": 1.0,  # full sign is always affected
        }


# ---------------------------------------------------------------------------
# Attack 4 — Graffiti / text overlay
# ---------------------------------------------------------------------------

class GraffitiAttack(BaseAttack):
    """
    Renders a short text string on the sign, simulating spray-painted graffiti.

    The text is placed at a random position within the sign ROI and drawn
    with a configurable colour, opacity, and font size.

    Config keys (all optional, defaults shown)
    ------------------------------------------
    texts           : ["X", "?", "//"]      — pool of strings to sample from
    font_size_ratio : [0.2, 0.5]            — font size as fraction of the
                                              shortest ROI dimension
    opacity_range   : [0.6, 0.95]           — text layer opacity
    palette         : ["#ffffff","#000000"] — text colour pool
    font_path       : null                  — absolute path to a .ttf font
                                              file; uses PIL default if null
    """

    name = "graffiti"

    def apply(
        self,
        image_path: Path,
        sample: Sample,
        rng: random.Random,
        config: dict[str, Any],
    ) -> tuple[Image.Image, dict[str, Any]]:
        image = _load_rgb(image_path)
        sign_mask = build_sign_mask(image.size, sample.roi, sample.shape)

        text = rng.choice(config.get("texts", ["X", "?", "//"]))
        opacity = pick_range(rng, config.get("opacity_range", [0.6, 0.95]))
        color = ImageColor.getrgb(rng.choice(config.get("palette", ["#ffffff", "#000000"])))

        min_side = min(sample.roi_x2 - sample.roi_x1, sample.roi_y2 - sample.roi_y1)
        font_size = max(8, int(min_side * pick_range(rng, config.get("font_size_ratio", [0.2, 0.5]))))

        # Load font — fall back to PIL built-in if no path is provided
        font_path = config.get("font_path")
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except (IOError, OSError):
            font = ImageFont.load_default()

        # Build a separate text mask so we can control opacity and clip to sign
        text_layer = Image.new("RGB", image.size, color)
        text_mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(text_mask)

        # Measure text so we don't place it out of bounds
        bbox = font.getbbox(text) if hasattr(font, "getbbox") else (0, 0, font_size * len(text), font_size)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        max_x = max(sample.roi_x1, sample.roi_x2 - text_w)
        max_y = max(sample.roi_y1, sample.roi_y2 - text_h)
        pos_x = rng.randint(sample.roi_x1, max_x)
        pos_y = rng.randint(sample.roi_y1, max_y)
        draw.text((pos_x, pos_y), text, fill=int(255 * opacity), font=font)

        # Clip text to sign shape only
        clipped_mask = ImageChops.multiply(text_mask, sign_mask)
        attacked = Image.composite(text_layer, image, clipped_mask)

        return attacked, {
            "attack": self.name,
            "text": text,
            "font_size": font_size,
            "position": [pos_x, pos_y],
            "opacity": round(opacity, 4),
            "color_rgb": list(color),
            "mask_fraction": round(mask_overlap_ratio(clipped_mask, sign_mask), 4),
        }


# ---------------------------------------------------------------------------
# Attack registry
# To add a new attack: instantiate it here.  Nothing else needs to change.
# ---------------------------------------------------------------------------

ATTACK_REGISTRY: dict[str, BaseAttack] = {
    attack.name: attack
    for attack in [
        ShadowAttack(),
        OcclusionAttack(),
        NoiseBlurAttack(),
        GraffitiAttack(),
    ]
}


# ---------------------------------------------------------------------------
# Physical transform (post-attack augmentation — not an attack itself)
# ---------------------------------------------------------------------------

def apply_physical_transform(
    image: Image.Image,
    rng: random.Random,
    config: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    """
    Optional post-attack augmentation simulating physical capture conditions
    (camera angle, lighting variation, motion blur).

    Controlled entirely by the ``physical_transform`` section of the YAML.
    Set ``enabled: false`` to skip entirely.

    Config keys (all optional, defaults shown)
    ------------------------------------------
    enabled               : false
    rotation_degrees      : [0.0, 0.0]
    brightness            : [1.0, 1.0]
    contrast              : [1.0, 1.0]
    gaussian_blur_radius  : [0.0, 0.0]
    """
    if not config.get("enabled", False):
        return image, {"enabled": False}

    rotation = pick_range(rng, config.get("rotation_degrees", [0.0, 0.0]))
    brightness = pick_range(rng, config.get("brightness", [1.0, 1.0]))
    contrast = pick_range(rng, config.get("contrast", [1.0, 1.0]))
    blur = pick_range(rng, config.get("gaussian_blur_radius", [0.0, 0.0]))

    out = image
    if abs(rotation) > 1e-6:
        out = out.rotate(rotation, resample=RESAMPLING_BICUBIC)
    if abs(brightness - 1.0) > 1e-6:
        out = ImageEnhance.Brightness(out).enhance(brightness)
    if abs(contrast - 1.0) > 1e-6:
        out = ImageEnhance.Contrast(out).enhance(contrast)
    if blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=blur))

    return out, {
        "enabled": True,
        "rotation_degrees": round(rotation, 4),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "gaussian_blur_radius": round(blur, 4),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def as_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True)


def relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def resolve_output_root(config: dict[str, Any], config_path: Path | None) -> Path:
    output_root = Path(config.get("output_root", "./physical_adv_attack/generated")).expanduser()
    if output_root.is_absolute() or config_path is None:
        return output_root.resolve()
    return (config_path.parent.parent / output_root).resolve()


def resolve_annotation_file(dataset_root: Path, config: dict[str, Any], split: str) -> Path:
    """Resolve the CSV annotation file path from config or fall back to convention."""
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


def load_samples(dataset_root: Path, config: dict[str, Any], split: str) -> list[Sample]:
    """Parse the annotation CSV and return a list of ``Sample`` objects."""
    annotation_path = resolve_annotation_file(dataset_root, config, split)
    shape_map = {int(k): str(v) for k, v in (config.get("shape_map") or {}).items()}

    with annotation_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        samples: list[Sample] = []
        for index, row in enumerate(reader):
            image_path = (dataset_root / row["Path"]).resolve()
            width = int(row.get("Width") or row.get("width") or 0)
            height = int(row.get("Height") or row.get("height") or 0)
            if width <= 0 or height <= 0:
                with Image.open(image_path) as img:
                    width, height = img.size

            class_id = int(row.get("ClassId") or row.get("class_id") or 0)
            samples.append(Sample(
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
            ))

    limit = config.get("limit")
    return samples[: int(limit)] if limit else samples


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

_LAST_PROGRESS_PERCENT = -1


def show_progress(done: int, total: int) -> None:
    """Print a compact ASCII progress bar to stdout (updates in-place)."""
    global _LAST_PROGRESS_PERCENT
    if total <= 0:
        return
    pct = int((done / total) * 100)
    if done not in (1, total) and pct == _LAST_PROGRESS_PERCENT:
        return
    _LAST_PROGRESS_PERCENT = pct
    filled = int(30 * done / total)
    bar = "#" * filled + "-" * (30 - filled)
    sys.stdout.write(f"\rProgress: [{bar}] {pct:3d}% ({done}/{total})")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(config: dict[str, Any], config_path: Path | None = None) -> dict[str, Any]:
    """
    Run the full attack generation pipeline.

    For each sample and each enabled attack, ``variants_per_image`` attacked
    images are produced and saved under::

        <output_root>/<split>/<attack_name>/<class_id:02d>/

    A ``manifest.csv`` is written to ``output_root`` containing one row per
    generated image with full reproducibility metadata.

    Parameters
    ----------
    config:
        Parsed YAML config dict (see ``config.yaml`` for the full schema).
    config_path:
        Path to the YAML file; used to resolve relative ``output_root``
        paths.  Pass ``None`` when calling programmatically with an absolute
        ``output_root``.

    Returns
    -------
    dict with keys: dataset_root, output_root, split,
    num_input_samples, num_generated_samples, manifest_path.
    """
    dataset_root = Path(config["dataset_root"]).expanduser().resolve()
    split = str(config.get("split", "test")).lower()
    output_root = resolve_output_root(config, config_path)
    ensure_dir(output_root)
    (output_root / ".keep").write_text("", encoding="utf-8")

    samples = load_samples(dataset_root, config, split)
    attacks_config: dict[str, Any] = config.get("attacks", {})
    seed = int(config.get("seed", 7))
    extension = str(config.get("output_extension", ".png"))
    attack_names = list(ATTACK_REGISTRY)

    # Count total outputs upfront for the progress bar
    total_outputs = sum(
        len(samples) * int(cfg.get("variants_per_image", 1))
        for name, cfg in attacks_config.items()
        if cfg.get("enabled", False) and name in ATTACK_REGISTRY
    )

    rows: list[dict[str, Any]] = []
    completed = 0

    for sample_index, sample in enumerate(samples):
        for attack_name, attack_config in attacks_config.items():
            if not attack_config.get("enabled", False):
                continue

            attack = ATTACK_REGISTRY.get(attack_name)
            if attack is None:
                print(f"\n[WARNING] Unknown attack '{attack_name}' — skipping. "
                      f"Available: {attack_names}")
                continue

            variant_count = int(attack_config.get("variants_per_image", 1))
            for variant_index in range(variant_count):
                # Unique seed per (sample, attack, variant) for full reproducibility
                attack_offset = attack_names.index(attack_name)
                rng = random.Random(seed + sample_index * 1000 + variant_index * 10 + attack_offset)

                attacked, attack_metadata = attack.apply(
                    sample.image_path, sample, rng, attack_config
                )
                transformed, transform_metadata = apply_physical_transform(
                    attacked, rng, config.get("physical_transform", {})
                )

                output_dir = output_root / sample.split / attack_name / f"{sample.class_id:02d}"
                ensure_dir(output_dir)
                output_path = output_dir / f"{sample.image_path.stem}_{attack_name}_{variant_index:02d}{extension}"
                transformed.save(output_path)

                rows.append({
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
                })

                completed += 1
                show_progress(completed, total_outputs)

    # Write manifest
    manifest_path = output_root / "manifest.csv"
    fieldnames = [
        "sample_id", "split", "class_id", "shape", "attack", "variant_index",
        "original_path", "output_path",
        "roi_x1", "roi_y1", "roi_x2", "roi_y2",
        "attack_parameters", "transform_parameters",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
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