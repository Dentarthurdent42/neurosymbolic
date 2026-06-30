"""
TRAP-Lite — end-to-end metacognitive AI pipeline on CIFAR-100-Hierarchical.

Week 4 capstone.  Each module maps to one TRAP formula:

    Perception   (f)          trap_lite.perception
    Reasoning    (g via EDCR) trap_lite.reasoning      f'(x; g(f(x), θ))
    Adaptation   (conformal)  trap_lite.adaptation     f'(x; g(f(x), θ))
    Transparency (LLM critic) trap_lite.transparency   g(f(x), θ)

Full pipeline: trap_lite.pipeline
One-command demo: python week4/demo.py --help
"""

from .pipeline import TRAPLitePipeline

__all__ = ['TRAPLitePipeline']
