import json
import numpy as np
import torch
import torch.nn as nn
import joblib
from ids.runtime.rules_v2 import run_rules

class MLPBinary(nn.Module):
    def __init__(self, in_dim, h1=256, h2=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

class MLPFamily(nn.Module):
    def __init__(self, in_dim, num_classes, h1=256, h2=128, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, num_classes)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        return self.out(x)

class IDSRuntimeV2:
    def __init__(
        self,
        binary_schema="schemas/v2/binary.json",
        binary_scaler="artifacts/v2/binary/scaler.pkl",
        binary_model="artifacts/v2/binary/model.pt",
        family_schema="schemas/v2/family.json",
        family_scaler="artifacts/v2/family/scaler.pkl",
        family_model="artifacts/v2/family/model.pt",
        label_encoder="datasets/v2/family/label_encoder.pkl",
        tau_low=0.2,
        tau_high=0.95,
        device=None,
    ):
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)

        # Cố gắng load mô hình từ inference_exp9
        import sys
        from pathlib import Path
        BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
        if str(BASE_DIR) not in sys.path:
            sys.path.append(str(BASE_DIR))

        try:
            from inference_exp9 import Exp9IDS
            model_path = BASE_DIR / "models/deploy_exp9"
            self.model = Exp9IDS(deploy_dir=str(model_path))
            print("[predictor_v2] Successfully loaded Exp9IDS as primary model.")
        except Exception as e:
            print(f"[predictor_v2] Could not load Exp9IDS: {e}. Running without ML (Rules only) or legacy mode.")
            self.model = None

    def predict_flow(self, flow: dict) -> dict:
        # -------- Rule (always run first for fast-path) --------
        hit = run_rules(flow)
        rule_result = None
        if hit is not None:
            rule_result = {
                "hit": True,
                "name": hit.name,
                "severity": hit.severity,
                "score": float(hit.score),
                "reason": hit.reason,
            }
        else:
            rule_result = {"hit": False}

        rule_high = (hit is not None and hit.severity == "high")
        rule_any = (hit is not None)

        if rule_high:
            return {
                "p_attack": 1.0,
                "stage": "rule",
                "verdict": "attack",
                "family": "rule_blocked",
                "family_conf": 1.0,
                "rule_info": {
                    "rule": hit.name,
                    "severity": hit.severity,
                    "score": float(hit.score),
                    "reason": hit.reason,
                },
                "rule_result": rule_result,
                "ml_binary": {"p_attack": 0.0, "verdict": "benign"},
                "ml_family": {"name": None, "conf": None},
                "final_source": "rule-only",
                "ml_verdict": "benign",
            }

        if not self.model:
            # Fallback to pure rules if ML is missing
            final_verdict = "suspicious" if rule_any else "benign"
            return {
                "p_attack": 0.0,
                "stage": "rule" if rule_any else "none",
                "verdict": final_verdict,
                "family": None,
                "family_conf": 0.0,
                "rule_info": {
                    "rule": hit.name, "severity": hit.severity, "score": float(hit.score), "reason": hit.reason
                } if hit else None,
                "rule_result": rule_result,
                "ml_binary": {"p_attack": 0.0, "verdict": "benign"},
                "ml_family": {"name": None, "conf": None},
                "final_source": "rule-only" if rule_any else "none",
                "ml_verdict": "benign",
            }

        # -------- ML (Exp9 Dual-Path Stacking) --------
        res = self.model.predict_single_dict(flow)
        is_attack = res.get("prediction", 0) == 1
        ml_verdict = "attack" if is_attack else "benign"
        ml_stage = res.get("stage", "exp9_ml")

        # Combine
        if rule_any and ml_verdict == "attack":
            final_verdict = "attack"
            final_source = "both"
            final_stage = ml_stage
        elif rule_any:
            final_verdict = "suspicious" if hit.severity == "medium" else "suspicious"
            final_source = "rule-only"
            final_stage = "rule"
        else:
            final_verdict = ml_verdict
            final_source = "ml-only" if ml_verdict != "benign" else "none"
            final_stage = ml_stage

        out = {
            "p_attack": 1.0 if is_attack else 0.0,
            "stage": final_stage,
            "verdict": final_verdict,
            "family": res.get("label", "benign") if final_verdict == "attack" else None,
            "family_conf": 1.0 if final_verdict == "attack" else 0.0,
            "rule_info": {
                 "rule": hit.name, "severity": hit.severity, "score": float(hit.score), "reason": hit.reason
            } if hit else None,
            "rule_result": rule_result,
            "ml_binary": {
                "p_attack": 1.0 if is_attack else 0.0,
                "verdict": ml_verdict,
            },
            "ml_family": {"name": None, "conf": None}, # obsolete but required by backward-compat
            "final_source": final_source,
            "ml_verdict": ml_verdict,
            "dede_error": res.get("error", 0.0),
        }
        return out