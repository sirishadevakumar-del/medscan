"""End-to-end inference: quality check -> modality routing -> ensemble + MC Dropout -> Grad-CAM -> decision."""
from pathlib import Path

import numpy as np
import torch

from .audit import AuditLog
from .data import build_transforms, load_any_image, to_rgb
from .gradcam import gradcam, overlay
from .models import load_checkpoint
from .policy import decide
from .quality import Thresholds, assess
from .uncertainty import mc_predict


class Analyzer:
    def __init__(self, ckpt, device=None, audit_path="runs/audit_log.jsonl"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ens, self.cfg, ck = load_checkpoint(ckpt, self.device)
        self.thresholds = {k: Thresholds(**v) for k, v in ck["thresholds"].items()}
        self.modalities = ck["modalities"]
        self.tasks = {k: v for k, v in ck["task_meta"].items() if k != "_modality"}
        self.tf = build_transforms(self.cfg.image_size, False)
        self.audit = AuditLog(audit_path)

    def route(self, x):
        """Predict imaging modality, return (modality, probs)."""
        with torch.no_grad():
            p = self.ens(x, "_modality")[0].cpu().numpy()
        return self.modalities[int(p.argmax())], p

    def analyze(self, image, task=None, patient_id=None, log=True):
        pil = load_any_image(image) if isinstance(image, (str, Path)) else image
        rgb = to_rgb(pil)
        x = self.tf(rgb).unsqueeze(0).to(self.device)
        modality, mod_probs = self.route(x)
        routed = None
        if task is None:
            cands = [t for t, v in self.tasks.items() if v["modality"] == modality]
            if not cands:
                raise ValueError(f"No task trained for predicted modality '{modality}'")
            task, routed = cands[0], cands
        meta = self.tasks[task]
        classes = meta["classes"]
        q = assess(rgb, self.thresholds[task])
        mc = mc_predict(self.ens, x, task, self.cfg.mc_samples, self.cfg.confidence_mode)
        probs = mc["probs"][0].cpu().numpy()
        conf = float(mc["confidence"][0])
        if meta["multilabel"]:
            target = int(probs.argmax())
            pred_label = [classes[i] for i in np.where(probs >= 0.5)[0]] or ["no finding above 0.5"]
        else:
            target = int(mc["pred"][0])
            pred_label = classes[target]
        cam, _ = gradcam(self.ens, x, task, target)
        ov = overlay(np.asarray(rgb.resize((self.cfg.image_size,) * 2)), cam)
        d = decide(conf, q["passed"], q["flags"], self.cfg.conf_high, self.cfg.conf_low)
        patient = self.audit.hash_id(patient_id)
        prior = self.audit.compare_with_prior(patient, task, probs.tolist())
        res = {"modality": modality, "modality_probs": dict(zip(self.modalities, map(float, mod_probs))),
               "task": task, "candidate_tasks": routed, "prediction": pred_label,
               "probabilities": dict(zip(classes, map(float, probs))),
               "uncertainty_std": dict(zip(classes, map(float, mc["std"][0].cpu().numpy()))),
               "confidence": conf, "mc_confidence": float(mc["mc_confidence"][0]),
               "tier": d["tier"], "action": d["action"], "notes": d["notes"],
               "quality": q, "longitudinal": prior, "gradcam_target": classes[target]}
        if log:
            res["case_id"] = self.audit.log(patient=patient, task=task, pred_label=pred_label, probs=probs.tolist(),
                                            confidence=conf, tier=d["tier"], quality_score=q["score"])
        return res, ov, cam
