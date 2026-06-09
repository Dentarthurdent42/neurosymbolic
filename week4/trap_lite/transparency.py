"""
Transparency module — TRAP formula: g(f(x), θ)

Reflexion-style LLM critic that receives:
    (feature_summary, model_prediction, conformal_set, fired_edcr_rules)
and produces a natural-language justification, or flags the case for human review.

Two backends supported:
    'claude'  — Anthropic Claude API   (default; requires ANTHROPIC_API_KEY)
    'local'   — Ollama local Llama-3   (requires ollama running locally)

Note from curriculum (Pitfall #5):
    Pure NL self-critique is sub-symbolic.  The fired EDCR rules are the
    typed symbolic representation that makes this genuinely metacognitive.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional


CRITIC_SYSTEM_PROMPT = """\
You are a metacognitive AI critic.  Your job is to assess whether a neural
classifier's prediction is trustworthy and to explain your reasoning.

You will be given:
1. A summary of the image features (top-5 softmax probabilities)
2. The classifier's prediction before symbolic correction
3. The classifier's prediction after EDCR symbolic rules were applied
4. The conformal prediction set (classes the system cannot rule out)
5. The EDCR rules that fired, if any

Your response must have exactly two parts:
VERDICT: TRUSTWORTHY | UNCERTAIN | FLAG_FOR_REVIEW
JUSTIFICATION: <one paragraph, ≤ 100 words>

Flag for review if: the conformal set has > 3 classes, EDCR changed the
prediction, or the top-2 probabilities differ by < 5 percentage points.
"""


@dataclass
class CriticOutput:
    verdict: str              # TRUSTWORTHY | UNCERTAIN | FLAG_FOR_REVIEW
    justification: str
    raw_response: str


class LLMCritic:
    """
    Calls an LLM to produce a natural-language critique of one prediction.

    Args:
        backend: 'claude' or 'local'
        model:   model name (e.g. 'claude-haiku-4-5-20251001' or 'llama3')
    """

    def __init__(
        self,
        backend: str = 'claude',
        model: str = 'claude-haiku-4-5-20251001',
    ):
        self.backend = backend
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            if self.backend == 'claude':
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=os.environ.get('ANTHROPIC_API_KEY')
                )
            # 'local' backend uses requests; no client needed
        return self._client

    def critique(
        self,
        top5_probs: list[tuple[int, float]],  # [(class_id, prob), ...]
        raw_pred: int,
        corrected_pred: int,
        conformal_set: set[int],
        fired_rules: list[str],
        class_names: Optional[list[str]] = None,
    ) -> CriticOutput:
        """
        Ask the LLM to evaluate one prediction.

        Args:
            top5_probs:     Top-5 (class_id, softmax_prob) pairs
            raw_pred:       Prediction before EDCR
            corrected_pred: Prediction after EDCR
            conformal_set:  Classes not ruled out by conformal predictor
            fired_rules:    EDCR rule strings that fired
            class_names:    Optional list of class name strings indexed by class ID
        """
        def name(c):
            if class_names and c < len(class_names):
                return f'{class_names[c]} ({c})'
            return str(c)

        top5_str = ', '.join(f'{name(c)}: {p:.2%}' for c, p in top5_probs)
        conf_str = '{' + ', '.join(name(c) for c in sorted(conformal_set)) + '}'
        rules_str = '; '.join(fired_rules) if fired_rules else 'none'

        user_msg = (
            f'Top-5 softmax: [{top5_str}]\n'
            f'Raw prediction:       {name(raw_pred)}\n'
            f'EDCR correction:      {name(corrected_pred)}'
            + (' (changed)' if corrected_pred != raw_pred else ' (unchanged)') + '\n'
            f'Conformal set:        {conf_str}\n'
            f'EDCR rules fired:     {rules_str}\n'
        )

        raw_response = self._call_llm(user_msg)
        return self._parse(raw_response)

    def _call_llm(self, user_msg: str) -> str:
        if self.backend == 'claude':
            client = self._get_client()
            msg = client.messages.create(
                model=self.model,
                max_tokens=256,
                system=CRITIC_SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': user_msg}],
            )
            return msg.content[0].text
        elif self.backend == 'local':
            import requests
            payload = {
                'model': self.model,
                'prompt': CRITIC_SYSTEM_PROMPT + '\n\n' + user_msg,
                'stream': False,
            }
            r = requests.post('http://localhost:11434/api/generate', json=payload, timeout=30)
            return r.json()['response']
        else:
            raise ValueError(f'Unknown backend: {self.backend}')

    @staticmethod
    def _parse(raw: str) -> CriticOutput:
        verdict = 'UNCERTAIN'
        justification = raw
        for line in raw.splitlines():
            if line.startswith('VERDICT:'):
                v = line.split(':', 1)[1].strip().upper()
                if v in ('TRUSTWORTHY', 'UNCERTAIN', 'FLAG_FOR_REVIEW'):
                    verdict = v
            elif line.startswith('JUSTIFICATION:'):
                justification = line.split(':', 1)[1].strip()
        return CriticOutput(verdict=verdict, justification=justification, raw_response=raw)
