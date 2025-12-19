"""
Experiment runner orchestrating preprocessing, model training, and attack execution.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import shutil

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = BASE_DIR / "results"
LOGS_DIR = RESULTS_ROOT / "logs"
METRICS_DIR = RESULTS_ROOT / "metrics"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = METRICS_DIR / "experiments_summary.csv"

# Import organized runner
try:
    from .organized_runner import create_run_folder, save_run_summary, create_run_index
    USE_ORGANIZED = True
except ImportError:
    USE_ORGANIZED = False
    print("⚠️  Warning: organized_runner not available, using legacy output")


def resolve_callable(path: str) -> Callable[..., Any]:
    module_name, attr = path.split(":")
    module = import_module(module_name)
    return getattr(module, attr)


def call_with_supported_args(func: Callable[..., Any], **kwargs):
    """Invoke callable with arguments filtered by its signature."""
    sig = inspect.signature(func)
    params = sig.parameters
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_var_kw:
        return func(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return func(**filtered)


def run_experiment(
    dataset_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    attack_cfg: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    dataset_name: Optional[str] = None,
    model_name: Optional[str] = None,
    attack_name: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute an experiment controlled by configuration dictionaries.

    Parameters
    ----------
    dataset_cfg : dict
        Configuration describing preprocessing callable and parameters.
    model_cfg : dict
        Configuration describing model training callable and parameters.
    attack_cfg : dict, optional
        Configuration for adversarial attack callable.
    output_dir : Path, optional
        Optional override for artefact location (not yet implemented).
    dataset_name / model_name / attack_name : str, optional
        Friendly identifiers used for logging.
    run_id : str, optional
        Custom run identifier. Defaults to timestamp if not provided.
    """

    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    preprocess_callable = dataset_cfg["callable"]
    if isinstance(preprocess_callable, str):
        preprocess_callable = resolve_callable(preprocess_callable)
    preprocess_params = dataset_cfg.get("params", {})
    preprocess_result = call_with_supported_args(preprocess_callable, **preprocess_params)

    processed_path = preprocess_result.data_path

    model_callable = model_cfg["callable"]
    if isinstance(model_callable, str):
        model_callable = resolve_callable(model_callable)
    model_params = model_cfg.get("params", {})
    model_outputs = call_with_supported_args(
        model_callable, processed_csv=processed_path, run_id=run_id, **model_params
    )

    attack_outputs = None
    if attack_cfg and attack_cfg.get("callable"):
        attack_callable = attack_cfg["callable"]
        if isinstance(attack_callable, str):
            attack_callable = resolve_callable(attack_callable)
        attack_params = attack_cfg.get("params", {})
        attack_outputs = call_with_supported_args(
            attack_callable,
            processed_csv=processed_path,
            model_outputs=model_outputs,
            run_id=run_id,
            **attack_params,
        )

    summary = {
        "run_id": run_id,
        "dataset": {
            "name": dataset_name,
            "callable": dataset_cfg.get("callable"),
            "params": preprocess_params,
            "processed_path": str(processed_path),
        },
        "model": {
            "name": model_name,
            "callable": model_cfg.get("callable"),
            "params": model_params,
            "outputs": _stringify_paths(model_outputs),
        },
        "attack": {
            "name": attack_name,
            "callable": attack_cfg.get("callable") if attack_cfg else None,
            "params": attack_cfg.get("params") if attack_cfg else None,
            "outputs": _stringify_paths(attack_outputs),
        },
        "output_dir": str(output_dir) if output_dir else None,
    }

    log_filename = _build_log_filename(
        dataset_name=dataset_name or "dataset",
        model_name=model_name or "model",
        attack_name=attack_name or "none",
        run_id=run_id,
    )
    log_path = LOGS_DIR / log_filename
    log_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    _append_summary_csv(
        run_id=run_id,
        dataset_name=dataset_name or "dataset",
        model_name=model_name or "model",
        attack_name=attack_name or "none",
        model_accuracy=model_outputs.get("accuracy") if isinstance(model_outputs, dict) else None,
        attack_accuracy=_extract_attack_accuracy(attack_outputs),
        processed_path=str(processed_path),
        model_path=str(model_outputs.get("model_path")) if isinstance(model_outputs, dict) else None,
        attack_adv_path=_extract_attack_path(attack_outputs),
    )

    # === ORGANIZED OUTPUT ===
    organized_folder = None
    if USE_ORGANIZED:
        try:
            # Create organized folder structure
            folders, timestamp = create_run_folder(
                dataset_name=dataset_name or "dataset",
                model_name=model_name or "model",
                attack_name=attack_name or "none"
            )
            organized_folder = folders['root']
            
            # Copy files to organized locations
            _organize_outputs(folders, model_outputs, attack_outputs, summary)
            
            # Save summary
            save_run_summary(folders, summary)
            
            # Update index
            create_run_index()
            
            print(f"\n📁 Results organized in: {organized_folder.relative_to(BASE_DIR)}/")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not organize outputs: {e}")

    return {
        "run_id": run_id,
        "processed_dataset": processed_path,
        "model_outputs": model_outputs,
        "attack_outputs": attack_outputs,
        "log_path": log_path,
        "organized_folder": organized_folder,
    }


def _stringify_paths(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _stringify_paths(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify_paths(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _build_log_filename(dataset_name: str, model_name: str, attack_name: str, run_id: str) -> str:
    safe_dataset = dataset_name.replace(" ", "_")
    safe_model = model_name.replace(" ", "_")
    safe_attack = attack_name.replace(" ", "_")
    return f"run_{safe_dataset}_{safe_model}_{safe_attack}_{run_id}.json"


def _extract_attack_accuracy(attack_outputs: Any) -> Optional[float]:
    if not isinstance(attack_outputs, dict):
        return None
    # prefer victim_adv accuracy if present, fallback to top-level accuracy
    if "victim_adv" in attack_outputs and isinstance(attack_outputs["victim_adv"], dict):
        return attack_outputs["victim_adv"].get("accuracy")
    return attack_outputs.get("accuracy")


def _extract_attack_path(attack_outputs: Any) -> Optional[str]:
    if not isinstance(attack_outputs, dict):
        return None
    for key in ("adversarial_samples", "adv_path"):
        if key in attack_outputs:
            value = attack_outputs[key]
            return str(value)
    return None


def _append_summary_csv(
    *,
    run_id: str,
    dataset_name: str,
    model_name: str,
    attack_name: str,
    model_accuracy: Any,
    attack_accuracy: Any,
    processed_path: str,
    model_path: Optional[str],
    attack_adv_path: Optional[str],
):
    """Append a lightweight summary row for quick comparison across runs."""
    import csv

    header = [
        "run_id",
        "dataset",
        "model",
        "attack",
        "model_accuracy",
        "attack_accuracy",
        "processed_dataset",
        "model_path",
        "adversarial_samples",
        "timestamp",
    ]
    row = {
        "run_id": run_id,
        "dataset": dataset_name,
        "model": model_name,
        "attack": attack_name,
        "model_accuracy": model_accuracy,
        "attack_accuracy": attack_accuracy,
        "processed_dataset": processed_path,
        "model_path": model_path,
        "adversarial_samples": attack_adv_path,
        "timestamp": datetime.now().isoformat(),
    }

    write_header = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _organize_outputs(folders, model_outputs, attack_outputs, summary):
    """Copy output files to organized folder structure."""
    if not isinstance(model_outputs, dict):
        return
    
    # Copy model files
    for key in ['model_path', 'metrics_path', 'report_path', 'confusion_matrix_path']:
        if key in model_outputs:
            src = Path(model_outputs[key])
            if src.exists():
                if 'model' in key:
                    dst = folders['models'] / src.name
                else:
                    dst = folders['metrics'] / src.name
                shutil.copy2(src, dst)
    
    # Copy attack files
    if isinstance(attack_outputs, dict):
        for key, value in attack_outputs.items():
            if isinstance(value, (str, Path)):
                src = Path(value)
                if src.exists() and src.is_file():
                    if 'adversarial' in key.lower() or 'adv' in key.lower():
                        dst = folders['adversarial'] / src.name
                    elif 'model' in key.lower():
                        dst = folders['models'] / src.name
                    else:
                        dst =folders['metrics'] / src.name
                    shutil.copy2(src, dst)
            
            # Handle nested dicts (e.g., attack.outputs.adversarial.metrics_path)
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (str, Path)):
                        src = Path(nested_value)
                        if src.exists() and src.is_file():
                            if 'adversarial' in nested_key.lower():
                                dst = folders['adversarial'] / src.name
                            elif 'model' in nested_key.lower():
                                dst = folders['models'] / src.name
                            else:
                                dst = folders['metrics'] / src.name
                            shutil.copy2(src, dst)


__all__ = ["run_experiment", "resolve_callable"]

