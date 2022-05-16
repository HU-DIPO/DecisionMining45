"""Tests for DM45_pruned.py."""
from typing import Tuple

import numpy as np

from decision_mining.core import c45
from decision_mining.core.dmn.models import DM45_pruned


def test_make_model(lin_med: Tuple[np.ndarray, np.ndarray]) -> None:
    """Tests the pipeline.make_model functions.

    Args:
        lin_med (Tuple[np.ndarray, np.ndarray]): Linearly separable medium\
            generated test set.
    """
    X, y = lin_med
    X = np.array([X, X]).T

    model = DM45_pruned.DM45_pruned().make_model(X, y, continuous_cols=np.array([0]))
    assert isinstance(
        model, c45.C45Classifier), f"Type should be c45.C45Classifier, not {type(model)}"
