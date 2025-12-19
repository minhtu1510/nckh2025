"""Command line interface for IDS research experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import yaml

from pipelines.training.runner import run_experiment

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "configs"


def load_yaml_config(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    datasets_cfg = load_yaml_config("datasets")
    models_cfg = load_yaml_config("models")
    attacks_cfg = load_yaml_config("attacks")

    parser = argparse.ArgumentParser(description="IDS adversarial experiment runner")
    parser.add_argument(
        "--dataset",
        required=False,
        choices=sorted(datasets_cfg.keys()),
        help="Dataset key defined in configs/datasets.yaml",
    )
    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        help="Python callable path for preprocessing, e.g. package.module:function",
    )
    parser.add_argument(
        "--model",
        required=False,
        choices=sorted(models_cfg.keys()),
        help="Model key defined in configs/models.yaml",
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        help="Python callable path for training, e.g. package.module:function",
    )
    parser.add_argument("--attack", default="none", choices=sorted(attacks_cfg.keys()))
    parser.add_argument("--run-id", dest="run_id", default=None, help="Optional run identifier override")
    parser.add_argument(
        "--model-params",
        default=None,
        help="Optional JSON string to override model params",
    )
    parser.add_argument(
        "--attack-params",
        default=None,
        help="Optional JSON string to override attack params",
    )
    parser.add_argument(
        "--dataset-params",
        default=None,
        help="Optional JSON string to override dataset preprocess params",
    )

    args = parser.parse_args()
    if not args.dataset and not args.dataset_path:
        parser.error("Either --dataset or --dataset-path must be provided.")
    if not args.model and not args.model_path:
        parser.error("Either --model or --model-path must be provided.")

    args.datasets_cfg = datasets_cfg
    args.models_cfg = models_cfg
    args.attacks_cfg = attacks_cfg
    return args


def merge_params(base: Dict[str, Any], override_json: str | None) -> Dict[str, Any]:
    params = dict(base or {})
    if override_json:
        overrides = json.loads(override_json)
        if not isinstance(overrides, dict):
            raise ValueError("Override parameters must be a JSON object")
        params.update(overrides)
    return params


def _infer_label(name: str | None, path: str | None, fallback: str) -> str:
    if name:
        return name
    if path:
        return path.split(":")[-1].split(".")[-1]
    return fallback


def _sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()


def _build_run_id(model_label: str, dataset_label: str, override: str | None = None) -> str:
    if override:
        return override
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{_sanitize_label(model_label)}_{_sanitize_label(dataset_label)}_{timestamp}"


def main() -> None:
    args = parse_args()

    dataset_cfg = args.datasets_cfg[args.dataset].copy() if args.dataset else {"callable": None, "params": {}}
    if args.dataset_path:
        dataset_cfg["callable"] = args.dataset_path

    model_cfg = args.models_cfg[args.model].copy() if args.model else {"callable": None, "params": {}}
    if args.model_path:
        model_cfg["callable"] = args.model_path

    dataset_name = _infer_label(args.dataset, args.dataset_path, "dataset")
    model_name = _infer_label(args.model, args.model_path, "model")
    attack_cfg = args.attacks_cfg.get(args.attack, {"callable": None, "params": {}}).copy()

    dataset_cfg["params"] = merge_params(dataset_cfg.get("params"), args.dataset_params)
    model_cfg["params"] = merge_params(model_cfg.get("params"), args.model_params)
    attack_cfg["params"] = merge_params(attack_cfg.get("params"), args.attack_params)

    run_id = _build_run_id(model_name, dataset_name, override=args.run_id)

    results = run_experiment(
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        attack_cfg=None if args.attack == "none" else attack_cfg,
        dataset_name=dataset_name,
        model_name=model_name,
        attack_name=args.attack,
        run_id=run_id,
    )

    print("\n=== Experiment Summary ===")
    print(f"Run ID: {results['run_id']}")
    print(f"Processed dataset: {results['processed_dataset']}")

    model_outputs = results.get("model_outputs")
    if model_outputs:
        print("Model outputs:")
        for key, value in model_outputs.items():
            print(f"  - {key}: {value}")

    attack_outputs = results.get("attack_outputs")
    if attack_outputs:
        print("Attack outputs:")
        for key, value in attack_outputs.items():
            print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
