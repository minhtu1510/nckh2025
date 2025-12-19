"""
Fast gradient and projected gradient attacks for tabular IDS datasets.

The attacker trains a linear surrogate (Logistic Regression) on the processed
dataset, crafts perturbations that push malicious traffic toward the benign
class, and optionally evaluates the effect on a victim model artefact produced
in step 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = BASE_DIR / "results"
METRICS_DIR = RESULTS_ROOT / "metrics"
ADV_DIR = RESULTS_ROOT / "adversarial_samples"

for directory in (METRICS_DIR, ADV_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _load_processed(
    processed_csv: Path, label_column: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(processed_csv)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {processed_csv}")
    feature_cols = [c for c in df.columns if c != label_column]
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_column].values.astype(int)
    return X, y, feature_cols


def _save_metrics(prefix: str, run_id: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Path]:
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    metrics_path = METRICS_DIR / f"{prefix}_metrics_{run_id}.csv"
    pd.DataFrame([{"accuracy": accuracy}]).to_csv(metrics_path, index=False)

    report_path = METRICS_DIR / f"{prefix}_classification_report_{run_id}.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)

    cm_path = METRICS_DIR / f"{prefix}_confusion_matrix_{run_id}.csv"
    pd.DataFrame(
        cm,
        index=[f"actual_{i}" for i in range(cm.shape[0])],
        columns=[f"pred_{i}" for i in range(cm.shape[1])],
    ).to_csv(cm_path)

    return {
        "accuracy": accuracy,
        "metrics_path": metrics_path,
        "report_path": report_path,
        "confusion_matrix_path": cm_path,
    }


def _load_victim(model_outputs: Optional[Dict[str, Any]]) -> Optional[Any]:
    """
    Best-effort load of a victim model artefact (joblib or Keras).
    Returns None if unavailable.
    """
    if not model_outputs:
        return None
    model_path = model_outputs.get("model_path")
    if not model_path:
        return None

    path = Path(model_path)
    if not path.exists():
        return None

    if path.suffix == ".pkl":
        return joblib.load(path)
    if path.suffix == ".h5":
        try:
            import tensorflow as tf
        except ImportError:
            return None
        return tf.keras.models.load_model(path)
    return None


def _predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    preds = model.predict(X)
    if preds.ndim == 1:
        return preds
    return preds[:, -1]


def _evaluate(model: Any, X: np.ndarray, y: np.ndarray, prefix: str, run_id: str) -> Dict[str, Any]:
    prob = _predict_proba(model, X)
    y_pred = (prob >= 0.5).astype(int)
    metrics = _save_metrics(prefix, run_id, y, y_pred)
    return {"accuracy": metrics["accuracy"], **metrics}


def _craft_fgsm(
    X: np.ndarray,
    y: np.ndarray,
    weight_vector: np.ndarray,
    epsilon: float,
    target_label: int,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    adv = X.copy()
    mask = y != target_label
    if not np.any(mask):
        return adv

    direction = np.sign(weight_vector)
    adv[mask] = adv[mask] - epsilon * direction
    adv[mask] = np.clip(adv[mask], clip_min, clip_max)
    return adv


def _craft_pgd(
    X: np.ndarray,
    y: np.ndarray,
    weight_vector: np.ndarray,
    epsilon: float,
    step_size: float,
    steps: int,
    target_label: int,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    adv = X.copy()
    base = X.copy()
    mask = y != target_label
    if not np.any(mask):
        return adv

    direction = np.sign(weight_vector)
    for _ in range(steps):
        adv[mask] = adv[mask] - step_size * direction
        delta = np.clip(adv[mask] - base[mask], -epsilon, epsilon)
        adv[mask] = np.clip(base[mask] + delta, clip_min, clip_max)
    return adv


def _attack_workflow(
    *,
    processed_csv: Path,
    run_id: str,
    model_outputs: Optional[Dict[str, Any]],
    label_column: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
    epsilon: float = 0.05,
    step_size: float = 0.01,
    pgd_steps: int = 10,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    mode: str = "fgsm",
) -> Dict[str, Any]:
    """
    Shared workflow: train surrogate, craft adversarial samples, evaluate victim/surrogate.
    """

    X, y, feature_cols = _load_processed(processed_csv, label_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    surrogate = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
        solver="liblinear",
    )
    surrogate.fit(X_train, y_train)

    baseline = _evaluate(surrogate, X_test, y_test, prefix=f"{mode}_baseline", run_id=run_id)

    weights = surrogate.coef_[0]
    if mode == "fgsm":
        X_adv = _craft_fgsm(
            X_test,
            y_test,
            weight_vector=weights,
            epsilon=epsilon,
            target_label=0,
            clip_min=clip_min,
            clip_max=clip_max,
        )
    else:
        X_adv = _craft_pgd(
            X_test,
            y_test,
            weight_vector=weights,
            epsilon=epsilon,
            step_size=step_size,
            steps=pgd_steps,
            target_label=0,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    adv_path = ADV_DIR / f"{mode}_adversarial_{run_id}.csv"
    adv_df = pd.DataFrame(X_adv, columns=feature_cols)
    adv_df[label_column] = y_test
    adv_df.to_csv(adv_path, index=False)

    surrogate_adv = _evaluate(surrogate, X_adv, y_test, prefix=f"{mode}_surrogate_adv", run_id=run_id)

    victim = _load_victim(model_outputs)
    victim_adv = None
    if victim is not None:
        victim_adv = _evaluate(victim, X_adv, y_test, prefix=f"{mode}_victim_adv", run_id=run_id)

    return {
        "run_id": run_id,
        "baseline": baseline,
        "surrogate_adv": surrogate_adv,
        "victim_adv": victim_adv,
        "adversarial_samples": adv_path,
    }


def run_fgsm_attack(
    *,
    processed_csv: Path,
    run_id: str,
    model_outputs: Optional[Dict[str, Any]] = None,
    label_column: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
    epsilon: float = 0.05,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> Dict[str, Any]:
    """Single-step FGSM-style attack using a linear surrogate."""
    return _attack_workflow(
        processed_csv=processed_csv,
        run_id=run_id,
        model_outputs=model_outputs,
        label_column=label_column,
        test_size=test_size,
        random_state=random_state,
        epsilon=epsilon,
        clip_min=clip_min,
        clip_max=clip_max,
        mode="fgsm",
    )


def run_pgd_attack(
    *,
    processed_csv: Path,
    run_id: str,
    model_outputs: Optional[Dict[str, Any]] = None,
    label_column: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
    epsilon: float = 0.05,
    step_size: float = 0.01,
    pgd_steps: int = 10,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> Dict[str, Any]:
    """Projected gradient descent (iterative FGSM) using a linear surrogate."""
    return _attack_workflow(
        processed_csv=processed_csv,
        run_id=run_id,
        model_outputs=model_outputs,
        label_column=label_column,
        test_size=test_size,
        random_state=random_state,
        epsilon=epsilon,
        step_size=step_size,
        pgd_steps=pgd_steps,
        clip_min=clip_min,
        clip_max=clip_max,
        mode="pgd",
    )


__all__ = ["run_fgsm_attack", "run_pgd_attack"]
