#!/usr/bin/env python
"""
TRAP-Lite one-command demo.

Usage examples:

    # Run on CIFAR-100 test split with default settings:
    python week4/demo.py

    # Use a trained backbone checkpoint:
    python week4/demo.py --backbone-ckpt week3/notebooks/resnet50_cifar100.pt

    # Use local Llama via Ollama instead of Claude:
    python week4/demo.py --llm-backend local --llm-model llama3

    # Adjust conformal coverage target:
    python week4/demo.py --alpha 0.05   # 95% coverage

    # Skip the LLM critic (faster):
    python week4/demo.py --no-critic
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

# Make sure the repo root is on the path when running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from week4.trap_lite.perception import PerceptionModel
from week4.trap_lite.reasoning import EDCRReasoner
from week4.trap_lite.adaptation import ConformalAdapter
from week4.trap_lite.transparency import LLMCritic
from week4.trap_lite.pipeline import TRAPLitePipeline


DATA_ROOT = Path('data')


def build_loaders(batch_size: int = 256):
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    full_train = torchvision.datasets.CIFAR100(DATA_ROOT, train=True,  download=True, transform=tfm)
    test_ds    = torchvision.datasets.CIFAR100(DATA_ROOT, train=False, download=True, transform=tfm)

    n_cal = 5000
    _, cal_ds = random_split(full_train, [len(full_train) - n_cal, n_cal],
                              generator=torch.Generator().manual_seed(42))

    cal_loader  = DataLoader(cal_ds, batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=2)
    return cal_loader, test_loader, test_ds.classes


def main():
    parser = argparse.ArgumentParser(
        description='TRAP-Lite demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--backbone',      default='resnet50', choices=['resnet50', 'dinov2'])
    parser.add_argument('--backbone-ckpt', type=Path, default=None,
                        help='Path to saved backbone state dict')
    parser.add_argument('--edcr-ckpt',    type=Path, default=None,
                        help='Path to pickled PyEDCR model')
    parser.add_argument('--alpha',         type=float, default=0.1,
                        help='Conformal miscoverage level (1-alpha = coverage)')
    parser.add_argument('--llm-backend',   default='claude', choices=['claude', 'local', 'none'])
    parser.add_argument('--llm-model',     default='claude-haiku-4-5-20251001')
    parser.add_argument('--no-critic',     action='store_true')
    parser.add_argument('--batch-size',    type=int, default=256)
    parser.add_argument('--device',        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print('\n' + '=' * 60)
    print('TRAP-Lite Demo')
    print('=' * 60)
    print(f'  Backbone:  {args.backbone}')
    print(f'  Alpha:     {args.alpha}  (target coverage {1 - args.alpha:.0%})')
    print(f'  LLM:       {"disabled" if args.no_critic else args.llm_backend}')
    print(f'  Device:    {args.device}')
    print('=' * 60)

    print('\n[1/4] Loading data…')
    cal_loader, test_loader, class_names = build_loaders(args.batch_size)

    print('[2/4] Building pipeline…')
    pipeline = TRAPLitePipeline.build(
        backbone=args.backbone,
        backbone_ckpt=args.backbone_ckpt,
        edcr_ckpt=args.edcr_ckpt,
        cal_loader=cal_loader,
        alpha=args.alpha,
        llm_backend=args.llm_backend if not args.no_critic else 'none',
        device=args.device,
    )
    pipeline.class_names = class_names

    print('[3/4] Running TRAP-Lite on test split…')
    result = pipeline.run_batch(
        test_loader,
        max_critic_samples=0 if args.no_critic else 5,
    )

    print('[4/4] Results:')
    TRAPLitePipeline.print_results_table(result)

    print('\nTRAP formula mapping:')
    print('  f(x)                   → Perception    (backbone forward pass)')
    print("  f'(x; g(f(x), θ))      → Reasoning     (EDCR rule correction)")
    print("  f'(x; g(f(x), θ))      → Adaptation    (conformal abstention)")
    print('  g(f(x), θ)             → Transparency  (LLM critic narrative)')


if __name__ == '__main__':
    main()
