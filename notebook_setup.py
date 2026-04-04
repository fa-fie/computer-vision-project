from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _first_existing_path(*candidates: Path) -> str | None:
    for path in candidates:
        if path.exists():
            return str(path.resolve())
    return None


def resolve_pipeline_weight_paths(
    project_root: str | Path,
    alexnet_weights_name: str = "adv_training_0.5_occlusion.pth",
) -> dict[str, str | None]:
    project_root = Path(project_root).resolve()
    model_dir = project_root / "model"

    alexnet_path = _first_existing_path(
        model_dir / alexnet_weights_name,
        model_dir / "adv_training_0.5_occlusion.pth",
        model_dir / "first_model_weights.pth",
    )
    attack_classifier_path = _first_existing_path(model_dir / "attack_classifier.pth")
    denoiser_path = _first_existing_path(model_dir / "denoiser.pth")
    e2e_path = _first_existing_path(model_dir / "pipeline_e2e.pth")

    missing = []
    if alexnet_path is None:
        missing.append("AlexNet weights")
    if attack_classifier_path is None:
        missing.append("AttackClassifier weights")
    if denoiser_path is None:
        missing.append("LearnedDenoiser weights")
    if missing:
        raise FileNotFoundError(
            "Missing required pretrained weights: " + ", ".join(missing)
        )

    return {
        "alexnet": alexnet_path,
        "attack_classifier": attack_classifier_path,
        "denoiser": denoiser_path,
        "pipeline_e2e": e2e_path,
    }


def _set_enabled_attack(config: dict[str, Any], attack_name: str) -> None:
    attacks = config.setdefault("attacks", {})
    for name, attack_cfg in attacks.items():
        if isinstance(attack_cfg, dict):
            attack_cfg["enabled"] = (name == attack_name)


def prepare_generator_config(
    config_path: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    attack_name: str,
    split: str = "train",
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    config["dataset_root"] = str(dataset_root)
    config["output_root"] = str(output_root)
    config["split"] = split
    config["dataset_type"] = config.get("dataset_type", "pytorch")
    config["annotation_file"] = config.get("annotation_file", "Train.csv")
    _set_enabled_attack(config, attack_name)

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    return config


def manifest_summary(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    df = pd.read_csv(manifest_path)
    return {
        "rows": int(len(df)),
        "splits": df["split"].value_counts().to_dict(),
        "attacks": df["attack"].value_counts().to_dict(),
        "classes": int(df["class_id"].nunique()),
    }


def manifest_has_attack_data(
    manifest_path: str | Path,
    attack_name: str,
    split: str = "train",
) -> bool:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        return False
    df = pd.read_csv(manifest_path)
    mask = (df["attack"] == attack_name) & (df["split"] == split)
    return bool(mask.any())


def ensure_attack_dataset(
    config_path: str | Path,
    dataset_root: str | Path,
    manifest_path: str | Path,
    attack_name: str = "occlusion",
    split: str = "train",
    force_regenerate: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    dataset_root = Path(dataset_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    output_root = manifest_path.parent

    prepare_generator_config(
        config_path=config_path,
        dataset_root=dataset_root,
        output_root=output_root,
        attack_name=attack_name,
        split=split,
    )

    generation_summary = None
    if force_regenerate or not manifest_has_attack_data(manifest_path, attack_name, split):
        from physical_adv_attack.generator import run_pipeline

        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        generation_summary = run_pipeline(config, config_path=config_path)

    if not manifest_has_attack_data(manifest_path, attack_name, split):
        raise FileNotFoundError(
            f"Could not prepare '{attack_name}' attack data at {manifest_path}"
        )

    return {
        "manifest_path": str(manifest_path),
        "generation_summary": generation_summary,
        "manifest_info": manifest_summary(manifest_path),
    }


def prepare_notebook_environment(
    project_root: str | Path,
    dataset_root: str | Path,
    manifest_path: str | Path,
    config_path: str | Path,
    attack_name: str = "occlusion",
    auto_prepare_data: bool = True,
    force_regenerate: bool = False,
    alexnet_weights_name: str = "adv_training_0.5_occlusion.pth",
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    config_path = Path(config_path).resolve()

    weight_paths = resolve_pipeline_weight_paths(
        project_root=project_root,
        alexnet_weights_name=alexnet_weights_name,
    )

    generation_summary = None
    if auto_prepare_data:
        prepared = ensure_attack_dataset(
            config_path=config_path,
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            attack_name=attack_name,
            split="train",
            force_regenerate=force_regenerate,
        )
        generation_summary = prepared["generation_summary"]
        manifest_info = prepared["manifest_info"]
    else:
        if not manifest_has_attack_data(manifest_path, attack_name, "train"):
            raise FileNotFoundError(
                f"Manifest missing '{attack_name}' attack data: {manifest_path}"
            )
        manifest_info = manifest_summary(manifest_path)

    return {
        "project_root": str(project_root),
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "config_path": str(config_path),
        "weight_paths": weight_paths,
        "generation_summary": generation_summary,
        "manifest_info": manifest_info,
    }


def format_generation_summary(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "Existing generated attack dataset will be reused."
    return json.dumps(summary, indent=2)
