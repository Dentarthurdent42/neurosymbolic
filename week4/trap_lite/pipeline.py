"""
TRAP-Lite pipeline — orchestrates all four TRAP components.

    f  (Perception)   →  g / f'  (Reasoning/Adaptation)  →  critic  (Transparency)

Usage:
    from trap_lite import TRAPLitePipeline
    pipeline = TRAPLitePipeline.build(backbone_ckpt='...', edcr_ckpt='...', cal_loader=...)
    result = pipeline.run_batch(images, labels)
    pipeline.print_results_table(result)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .perception import PerceptionModel
from .reasoning import EDCRReasoner, FINE_TO_SUPER
from .adaptation import ConformalAdapter
from .transparency import LLMCritic, CriticOutput


@dataclass
class PipelineResult:
    # Per-sample arrays (N,)
    labels: np.ndarray
    raw_preds: np.ndarray
    edcr_preds: np.ndarray
    conformal_sets: list[set[int]]
    abstain: np.ndarray
    error_flags: np.ndarray
    fired_rules: list[list[str]]

    # Scalar metrics
    acc_f_only: float = 0.0
    acc_f_edcr: float = 0.0
    selective_acc_90: float = 0.0
    selective_acc_95: float = 0.0
    empirical_coverage: float = 0.0

    # LLM critic outputs for a sample of flagged cases
    critic_samples: list[dict] = field(default_factory=list)


class TRAPLitePipeline:
    """
    End-to-end TRAP-Lite pipeline.

    Build it from trained artefacts, then call run_batch() on a DataLoader.
    """

    def __init__(
        self,
        perception: PerceptionModel,
        reasoner: EDCRReasoner,
        adapter: ConformalAdapter,
        critic: Optional[LLMCritic] = None,
        class_names: Optional[list[str]] = None,
    ):
        self.perception = perception
        self.reasoner = reasoner
        self.adapter = adapter
        self.critic = critic
        self.class_names = class_names

    @classmethod
    def build(
        cls,
        backbone: str = 'resnet50',
        backbone_ckpt: Optional[Path] = None,
        edcr_ckpt: Optional[Path] = None,
        cal_loader: Optional[DataLoader] = None,
        alpha: float = 0.1,
        llm_backend: str = 'claude',
        device: str = 'cpu',
    ) -> 'TRAPLitePipeline':
        """
        Convenience constructor.

        Args:
            backbone:      'resnet50' or 'dinov2'
            backbone_ckpt: path to saved ResNet-50 state dict
            edcr_ckpt:     path to pickled PyEDCR model
            cal_loader:    DataLoader for conformal calibration
            alpha:         conformal miscoverage level (default 0.1 → 90% coverage)
            llm_backend:   'claude' or 'local'
            device:        torch device string
        """
        perception = PerceptionModel(backbone, checkpoint=backbone_ckpt, device=device)

        if edcr_ckpt and edcr_ckpt.exists():
            reasoner = EDCRReasoner.from_checkpoint(edcr_ckpt)
        else:
            reasoner = EDCRReasoner()

        adapter = ConformalAdapter(alpha=alpha)
        if cal_loader is not None:
            proba_cal, labels_cal = _collect_proba(perception, cal_loader)
            adapter.fit(proba_cal, labels_cal)

        critic = LLMCritic(backend=llm_backend)

        return cls(perception, reasoner, adapter, critic)

    @torch.no_grad()
    def run_batch(
        self,
        loader: DataLoader,
        max_critic_samples: int = 5,
    ) -> PipelineResult:
        """Run all TRAP components on a DataLoader; return aggregated results."""
        all_labels, all_raw_preds, all_probs = [], [], []

        for images, labels in loader:
            out = self.perception(images)
            all_labels.append(labels.numpy())
            all_raw_preds.append(out.preds.numpy())
            all_probs.append(out.probs.numpy())

        labels = np.concatenate(all_labels)
        raw_preds = np.concatenate(all_raw_preds)
        probs = np.concatenate(all_probs)

        # Reasoning: EDCR rule application
        reasoning_out = self.reasoner(raw_preds)

        # Adaptation: conformal prediction sets
        adaptation_out = self.adapter.predict(probs)

        # Metrics
        acc_f = (raw_preds == labels).mean()
        acc_edcr = (reasoning_out.corrected_preds == labels).mean()
        metrics_90 = self.adapter.selective_accuracy(probs, labels)

        # Transparency: critique flagged samples (subsample for cost)
        critic_samples = []
        if self.critic is not None:
            flagged_idx = np.where(reasoning_out.error_flags | adaptation_out.abstain)[0]
            sampled = flagged_idx[:max_critic_samples]
            for i in sampled:
                top5 = sorted(enumerate(probs[i].tolist()), key=lambda x: -x[1])[:5]
                c_out = self.critic.critique(
                    top5_probs=top5,
                    raw_pred=int(raw_preds[i]),
                    corrected_pred=int(reasoning_out.corrected_preds[i]),
                    conformal_set=adaptation_out.prediction_sets[i],
                    fired_rules=reasoning_out.fired_rules[i],
                    class_names=self.class_names,
                )
                critic_samples.append({
                    'idx': int(i),
                    'true_label': int(labels[i]),
                    'verdict': c_out.verdict,
                    'justification': c_out.justification,
                })

        return PipelineResult(
            labels=labels,
            raw_preds=raw_preds,
            edcr_preds=reasoning_out.corrected_preds,
            conformal_sets=adaptation_out.prediction_sets,
            abstain=adaptation_out.abstain,
            error_flags=reasoning_out.error_flags,
            fired_rules=reasoning_out.fired_rules,
            acc_f_only=float(acc_f),
            acc_f_edcr=float(acc_edcr),
            selective_acc_90=metrics_90['selective_accuracy'],
            empirical_coverage=metrics_90['empirical_coverage'],
            critic_samples=critic_samples,
        )

    @staticmethod
    def print_results_table(result: PipelineResult) -> None:
        """Print the Week 4 capstone results table."""
        print('\n' + '=' * 60)
        print('TRAP-Lite Results')
        print('=' * 60)
        print(f'  f only           accuracy : {result.acc_f_only:.4f}')
        print(f'  f + EDCR         accuracy : {result.acc_f_edcr:.4f}  '
              f'(Δ={result.acc_f_edcr - result.acc_f_only:+.4f})')
        print(f'  conformal cov.   empirical: {result.empirical_coverage:.4f}')
        print(f'  selective acc.   @ 90% cov: {result.selective_acc_90:.4f}')
        print(f'  abstention rate           : {result.abstain.mean():.4f}')
        print(f'  EDCR error flags          : {result.error_flags.mean():.4f}')
        if result.critic_samples:
            print('\nLLM critic — flagged samples:')
            for s in result.critic_samples:
                print(f"  [{s['verdict']:>18s}] idx={s['idx']} "
                      f"true={s['true_label']}  →  {s['justification'][:80]}…")
        print('=' * 60)


def _collect_proba(
    perception: PerceptionModel,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect softmax probabilities and labels from a DataLoader."""
    proba_list, label_list = [], []
    for images, labels in loader:
        out = perception(images)
        proba_list.append(out.probs.numpy())
        label_list.append(labels.numpy())
    return np.concatenate(proba_list), np.concatenate(label_list)
