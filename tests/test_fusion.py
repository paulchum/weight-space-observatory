import math

import pytest
import torch

from wso.metrics.fusion import calculate_fusion_entropy, fusion_probabilities


def test_fusion_probabilities_sum_to_one() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0])
    probs = fusion_probabilities(logits)
    assert float(probs.sum().item()) == pytest.approx(1.0)


def test_uniform_entropy() -> None:
    logits = torch.zeros(4)
    entropy = calculate_fusion_entropy(logits)
    assert entropy == pytest.approx(math.log(4), abs=1e-6)
