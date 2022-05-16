"""Tests for DM45.py."""
from typing import List, Tuple

import numpy as np
import pytest

from decision_mining.core import c45
from decision_mining.core.dmn.rule import Rule
from decision_mining.core.dmn.models import DM45


@pytest.fixture
def trained_c45() -> c45.C45Classifier:
    """Trained C4.5 model.

    Class is 1 if: 50 < X[:, 0] < 80 and X[:, 1] == Union[1, 2]
    Else, Class is 0

    Returns:
        c45.C45Classifier: Trained C4.5 model.
    """
    clss = c45.C45Classifier(continuous_cols=np.array([0, ]))
    X = np.array([np.arange(100), np.zeros(100)]).T
    X[:, 1] = X[:, 0] % 3
    y = np.logical_and(np.logical_and(X[:, 0] > 50, X[:, 1] > 0), X[:, 0] < 80).astype(np.int32)
    clss.fit(X, y)

    return clss


@pytest.fixture
def c45_paths() -> List[List[Tuple[c45._TreeNode, int, float, bool]]]:
    """Generate rule path based on C4.5.

    Returns:
        List[Tuple[c45._TreeNode, int, float, bool]]: List of tuples representing rules.
    """
    clss = c45.C45Classifier(continuous_cols=np.array([0, ]))
    X = np.array([np.arange(100), np.zeros(100)]).T
    X[:, 1] = X[:, 0] % 3
    y = np.logical_and(np.logical_and(X[:, 0] > 50, X[:, 1] > 0), X[:, 0] < 80).astype(np.int32)
    clss.fit(X, y)
    return list(c45.traverse_c45(clss))


def test_make_model(lin_med: Tuple[np.ndarray, np.ndarray]) -> None:
    """Tests the pipeline.make_model functions.

    Args:
        lin_med (Tuple[np.ndarray, np.ndarray]): Linearly separable medium\
            generated test set.
    """
    X, y = lin_med
    X = np.array([X, X]).T

    model = DM45.DM45().make_model(X, y, continuous_cols=np.array([0]))
    assert isinstance(
        model, c45.C45Classifier), f"Type should be c45.C45Classifier, not {type(model)}"


def test_extract_rules(trained_c45: c45.C45Classifier) -> None:
    """Tests pipeline.extract_rules function.

    As the underlying functions have already been tested fully, we're only checking typing.

    Args:
        trained_c45 (c45.C45Classifier): Fitted C45Classifier.
        trained_fuzzy (fuzzy.FuzzyClassifier): Fitted FuzzyClassifier.
    """
    rules = DM45.DM45().extract_rules([0, 1], trained_c45)
    assert isinstance(rules, list), f"Should be list, not {type(rules)}"
    assert all(isinstance(rule, Rule) for rule in rules)


def test_make_rule(c45_paths: List[List[Tuple[c45._TreeNode, int, float, bool]]]) -> None:
    """Tests the make_rule function.

    Args:
        c45_paths (List[List[Tuple[c45._TreeNode, int, float, bool]]]): Generated rule paths
    """
    simple_path: List[Tuple[c45._TreeNode, int, float, bool]] = c45_paths[0]
    simple_rule: Rule = DM45.DM45().make_rule(lambda: Rule([0, 1]), simple_path)
    simple_expect = [{"threshold": 51.5, "<": True}]

    assert simple_rule.decision == 0, f"Decision should be 0, not {simple_rule.decision}"
    assert simple_rule.cols[0] == simple_expect, f"Column 0 \
should be {simple_expect}, not {simple_rule.cols[0]}"
    assert simple_rule.cols[1] is None, f"Column 1 should be None, not {simple_rule.cols[1]}"

    # Test more >1 continuous split
    cont_path: List[Tuple[c45._TreeNode, int, float, bool]] = c45_paths[4]
    cont_rule: Rule = DM45.DM45().make_rule(lambda: Rule([0, 1]), cont_path)
    complex_expect = [{"threshold": 51.5, "<": False},
                      {"threshold": 79.5, "<": False}]

    assert cont_rule.decision == 0, f"Decision should be 0, not {cont_rule.decision}"
    assert cont_rule.cols[0] == complex_expect, f"Column 0 should be {complex_expect}, \
not {cont_rule.cols[0]}"
    assert cont_rule.cols[1] is None, f"Column 1 should be None, not {cont_rule.decision}"

    # Test complex split with two continuous splits and one categorical
    complex_path: List[Tuple[c45._TreeNode, int, float, bool]] = c45_paths[2]
    complex_rule: Rule = DM45.DM45().make_rule(lambda: Rule([0, 1]), complex_path)
    cat_complex = 1.0
    cont_complex = [{"threshold": 51.5, "<": False},
                    {"threshold": 79.5, "<": True}]
    assert complex_rule.decision == 1, f"Decision should be 1, not {complex_rule.decision}"
    assert complex_rule.cols[0] == cont_complex, f"Column 0 should be {cont_complex}, \
not {complex_rule.cols[0]}"
    assert complex_rule.cols[1] == cat_complex, f"Column 1 should be {cat_complex}, \
not {complex_rule.cols[1]}"


def test_make_rules(c45_paths: List[List[Tuple[c45._TreeNode, int, float, bool]]]) -> None:
    """Tests the make_rules.

    Args:
        c45_paths (List[List[Tuple[c45._TreeNode, int, float, bool]]]): Generated rule paths
    """
    rules = DM45.DM45().make_rules([0, 1], c45_paths)

    assert len(rules) == 5, f"Should return 5 rules, not {len(rules)}"
    assert all(isinstance(rule_, Rule) for rule_ in rules), "All rules should be of type Rule"

    complex_rule: Rule = rules[2]
    cat_complex = 1.0
    cont_complex = [{"threshold": 51.5, "<": False},
                    {"threshold": 79.5, "<": True}]
    assert complex_rule.decision == 1, f"Decision should be 1, not {complex_rule.decision}"
    assert complex_rule.cols[0] == cont_complex, f"Column 0 should be {cont_complex}, \
not {complex_rule.cols[0]}"
    assert complex_rule.cols[1] == cat_complex, f"Column 1 should be {cat_complex}, \
not {complex_rule.cols[1]}"
