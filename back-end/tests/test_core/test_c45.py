"""Testing c45.py."""
from typing import Tuple, List

import numpy as np
import pytest
from more_itertools import collapse

from decision_mining.core import c45


@pytest.fixture
def clsfr() -> c45.C45Classifier:
    """Basic C45Classifier for testing. A new one is generated for each test.

    Yields:
        c45.C45Classifier: Fresh C45Classifier for testing.
    """
    classifier = c45.C45Classifier()
    yield classifier


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


def test_c45_predict(clsfr: c45.C45Classifier,
                     lin_med: Tuple[np.ndarray, np.ndarray]) -> None:
    """Testing C45Classifier's predict function.

    Args:
        clsfr (c45.C45Classifier): Fresh C45Classifier for testing.
        lin_med (Tuple[np.ndarray, np.ndarray]): Linearly separable medium\
            generated test set.
    """
    X, y = lin_med
    clsfr.fit(X, y)
    assert [clsfr.predict(np.array([n])) for n in range(5)] == [1, 0, 3, 4, 5]
    X = np.stack((X, X), axis=1)
    clsfr.fit(X, y)
    assert [clsfr.predict(np.array((n, n)))
            for n in range(5)] == [1, 0, 3, 4, 5]

    X: np.ndarray = np.arange(0, 5)
    X: np.ndarray = np.stack((X, X), axis=1)
    assert (clsfr.predict(X) == np.array([1, 0, 3, 4, 5])).all()
    X = np.append(X, np.zeros(X.shape, dtype=np.int32), axis=1)
    with pytest.raises(ValueError, match="Expected X.shape to be"):
        clsfr.predict(X)

    with pytest.raises(ValueError, match="Expected X.shape to be"):
        clsfr.predict(np.stack((X, X), axis=1))

    with pytest.raises(TypeError, match="Expected X to be np.ndarray with"):
        clsfr.predict(0)
    with pytest.raises(TypeError, match="Expected X to be np.ndarray with"):
        clsfr.predict(X.astype(object))

    # Testing with continuous data
    clsfr = c45.C45Classifier(continuous_cols=np.array([0, ]))
    X = np.array([np.arange(100), np.zeros(100)]).T
    y = (X[:, 0] > 50).astype(np.int32)
    clsfr.fit(X, y)
    y_pred = clsfr.predict(np.array([[42., 0.], [67., 0.]]))
    y_true = np.array([0, 1])
    assert (y_pred == y_true).all(), f"{y_pred} should equal {y_true}"


def test_c45_bc1(clsfr: c45.C45Classifier,
                 med_set: Tuple[np.ndarray, np.ndarray]) -> None:
    """Testing C45Classifier's base case #1.

    Args:
        clsfr (c45.C45Classifier): Fresh C45Classifier for testing.
        med_set (Tuple[np.ndarray, np.ndarray]): Medium generated test set.
    """
    X, y = med_set
    # Test base case 1
    mask = y == 1
    Xb1 = X[mask]
    yb1 = y[mask]
    clsfr.fit(Xb1, yb1)
    assert clsfr.node.is_leaf
    assert clsfr.node.pred_class == 1


def test_c45_min_objs(clsfr: c45.C45Classifier,
                      med_set: Tuple[np.ndarray, np.ndarray]) -> None:
    """Testing C45Classifier's MIN_OBJS feature.

    Args:
        clsfr (c45.C45Classifier): Fresh C45Classifier for testing.
        med_set (Tuple[np.ndarray, np.ndarray]): Medium generated test set.
    """
    X = np.array([0])
    y = np.array([1])
    clsfr.fit(X, y)
    assert clsfr.node.is_leaf
    assert clsfr.node.pred_class == 1

    y, X = med_set
    X = X[:9]
    y = y[:9]
    print(X, y)
    clsfr = c45.C45Classifier(min_objs=10)
    clsfr.fit(X, y)
    assert clsfr.node.is_leaf
    assert clsfr.node.pred_class == "Ja"


def test_c45_bc2(clsfr: c45.C45Classifier,
                 small_set: Tuple[np.ndarray, np.ndarray]) -> None:
    """Testing C45Classifier's base case #2.

    Args:
        clsfr (c45.C45Classifier): Fresh C45Classifier for testing.
        small_set (Tuple[np.ndarray, np.ndarray]): Small generated test set.
    """
    X, y = small_set
    # Test base case 2
    clsfr.fit(X, y)
    top_node = clsfr.node
    assert not top_node.is_leaf, "top_node should not be a leaf"
    bottom_node = top_node.nodes[1.]
    assert bottom_node.is_leaf, "bottom_node should be leaf"
    assert (bottom_node.gain_ratios == np.array([0., 0.])).all(), f"Gain\
 ratios should be [0., 0.], not {bottom_node.gain_ratios}"


def test_c45_bc3(clsfr: c45.C45Classifier,
                 med_set: Tuple[np.ndarray, np.ndarray]) -> None:
    """Testing C45Classifier's base case #3.

    Args:
        clsfr (c45.C45Classifier): Fresh C45Classifier for testing.
        med_set (Tuple[np.ndarray, np.ndarray]): Medium generated test set.
    """
    X, y = med_set
    clsfr.fit(X, y)
    # Test base case 2
    assert clsfr.predict(np.array(["?"], dtype=object)) == 1


def test_c45_fit(clsfr: c45.C45Classifier,
                 lin_small: Tuple[np.ndarray, np.ndarray]) -> None:
    """Testing C45Classifier's fit function.

    Args:
        clsfr (c45.C45Classifier): Fresh C45Classifier for testing.
        lin_small (Tuple[np.ndarray, np.ndarray]): Small generated test set.
    """
    X, y = lin_small
    clsfr.fit(X, y)


def test_find_best_split(clsfr: c45.C45Classifier,
                         lin_med: Tuple[np.ndarray, np.ndarray]) -> None:
    """Testing C45Classifier's __TreeNode's best_split function.

    Args:
        clsfr (c45.C45Classifier): Fresh C45Classifier for testing.
        lin_med (Tuple[np.ndarray, np.ndarray]): Linearly separable medium\
            generated test set.
    """
    X, y = lin_med
    clsfr.fit(X, y)
    node = clsfr.node
    assert node.gain_ratios[0] == 1., f"`gain_ratios` should be [1.], not\
        {node.gain_ratios}"
    assert node.attribute == 0, f"`attribute` should 0, not {node.attribute}"


def test_find_best_split_continuous() -> None:
    """Testing C45Classifier's __TreeNode's best_split function with continuous values."""
    # Test with continuous data
    X = np.array([np.arange(100), np.zeros(100)]).T
    y = (X[:, 0] > 50).astype(np.int32)

    clsfr = c45.C45Classifier(continuous_cols=np.array([0, ]))
    clsfr.fit(X, y)
    threshold = 50.5
    gain = 1.
    node = clsfr.node
    assert node.gain_ratios[0] == gain, f"`gain_ratios` should be [{gain}, ...], not\
        {node.gain_ratios}"
    assert node.attribute == 0, f"`attribute` should be 0, not {node.attribute}"
    assert node.threshold == threshold, f"`threshold` should be 50.5, not {node.threshold}"


def test_gain(lin_small: Tuple[np.ndarray, np.ndarray],
              med_set: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test the c45.gain function.

    Args:
        lin_small (Tuple[np.ndarray, np.ndarray]): Small generated test set.
        med_set (Tuple[np.ndarray, np.ndarray]): Medium generated test set.
    """
    X, y = lin_small
    mask = X[:, 0] == 0
    S = X[mask][:, 0]
    C = y[mask]
    assert abs((g := c45.gain(S, C)) - (t := 0.)) < 1e-3, f"{g=}, expected {t}"
    X, y = med_set
    assert abs((g := c45.gain(X, y)) - (t := 0.05977)
               ) < 1e-3, f"{g=}, expected {t}"


def test_split_info(med_set: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test the c.45 split_info function.

    Args:
        med_set (Tuple[np.ndarray, np.ndarray]): Medium generated test set.
    """
    X, y = med_set

    assert abs((g := c45.split_info(X, y)) - (t := 0.971)
               ) < 1e-3, f"{g=}, expected {t}"


def test_gain_ratio(med_set: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test the c.45 split_info function.

    Args:
        med_set (Tuple[np.ndarray, np.ndarray]): Medium generated test set.
    """
    X, y = med_set

    assert abs((g := c45.gain_ratio(X, y)) - (t := 0.0616)
               ) < 1e-3, f"{g=}, expected {t}"
    X = np.zeros(y.shape)
    assert (g := c45.gain_ratio(X, y)) == (t := 0.), f"{g=}, expected {t}"


def test_find_threshold() -> None:
    """Tests the c45.find_threshold function."""
    X = np.arange(100)
    y = (X > 50).astype(np.int32)

    threshold = 50.5
    gain = 1.

    thresh_test, gain_test = c45.find_threshold(X, y)
    assert thresh_test == threshold, f"Expected threshold to be {threshold}, not {thresh_test}"
    assert abs(gain - gain_test) < 1e-3, f"Expected gain to be {gain}, not {gain_test}"

    y = (X > 75).astype(np.int32)
    threshold = 75.5
    thresh_test, gain_test = c45.find_threshold(X, y)
    assert thresh_test == threshold, f"Expected threshold to be {threshold}, not {thresh_test}"
    assert abs(gain - gain_test) < 1e-3, f"Expected gain to be {gain}, not {gain_test}"


def test_traverse_node(trained_c45: c45.C45Classifier) -> None:
    """Tests the rule_c45.traverse_node function.

    Args:
        trained_c45 (c45.C45Classifier): Fixture with trained C4.5 model.
    """
    top_node = trained_c45.node
    leaf_node_1: c45._TreeNode = top_node.nodes[True]
    assert leaf_node_1.is_leaf, "This node should be leaf, check c45.py"
    assert c45.traverse_node(leaf_node_1, tuple()) == ((leaf_node_1, leaf_node_1.pred_class),)

    rules: List[c45._TreeNode] = list(collapse(c45.traverse_node(top_node, tuple()),
                                               base_type=tuple))
    branch_node: c45._TreeNode = top_node.nodes[False]
    leaf_node_2: c45._TreeNode = branch_node.nodes[False]

    assert leaf_node_2.is_leaf, "This node should be leaf, check c45.py"

    print(rules[-1])
    assert rules[-1] == ((top_node, top_node.attribute, top_node.threshold, False),
                         (branch_node, branch_node.attribute, branch_node.threshold, False),
                         (leaf_node_2, leaf_node_2.pred_class))


def test_traverse_c45(trained_c45: c45.C45Classifier) -> None:
    """Tests the c45.traverse_c45.

    Args:
        trained_c45 (c45.C45Classifier): Fixture with trained C4.5 model.
    """
    top_node = trained_c45.node
    branch_node: c45._TreeNode = top_node.nodes[False]
    leaf_node_2: c45._TreeNode = branch_node.nodes[False]
    assert list(c45.traverse_c45(trained_c45))[-1] == ((top_node, top_node.attribute,
                                                        top_node.threshold, False),
                                                       (branch_node, branch_node.attribute,
                                                        branch_node.threshold, False),
                                                       (leaf_node_2, leaf_node_2.pred_class))


def test_reduced_error_pruning(large_set: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test c45.reduced_error_pruning.

    Args:
        large_set (Tuple[np.ndarray, np.ndarray]): Large pregenerated dataset
    """
    X, y = large_set
    X_val = np.array([[28.673077], [11.780550], [8.930731], [48.770002],
                      [41.626866], [33.604472], [24.512377], [38.155715],
                      [42.148802], [9.149209], [29.184731], [47.135539],
                      [40.216283], [20.819470], [45.978458], [47.412796],
                      [34.578815], [50.378198], [17.575258], [23.550003],
                      [40.532929], [5.453591], [36.329400], [21.984428],
                      [22.882816], [24.071271], [24.736609], [19.325776],
                      [36.384932], [32.440150], [9.301599], [15.938148],
                      [25.814323], [34.870604], [47.042858], [34.422533],
                      [34.112827], [35.117239], [35.819587], [18.966714],
                      [21.807965], [40.910757], [15.025889], [14.494765],
                      [39.944821], [50.020695], [12.452817], [5.680780],
                      [33.076851], [24.804678], [33.748685], [47.257410],
                      [7.119658], [39.090064], [35.564417], [28.956290],
                      [33.849482], [28.307994], [49.400517], [23.071627]])
    y_val = np.array(['warm', 'cold', 'cold', 'hothot', 'hot', 'hot', 'warm', 'hothot',
                      'hothot', 'cold', 'warm', 'hothot', 'hothot', 'warm', 'hothot', 'hothot',
                      'warm', 'hothot', 'warm', 'warm', 'hot', 'cold', 'hot', 'warm',
                      'warm', 'warm', 'warm', 'warm', 'hot', 'hot', 'cold', 'cold',
                      'warm', 'warm', 'hothot', 'hot', 'hot', 'hot', 'hothot', 'warm',
                      'warm', 'hothot', 'cold', 'cold', 'hot', 'hothot', 'cold', 'cold',
                      'hot', 'warm', 'hot', 'hothot', 'cold', 'hot', 'hot', 'warm',
                      'hot', 'warm', 'hothot', 'warm'])
    clsfr = c45.C45Classifier([0])
    clsfr.fit(X, y)
    clsfr.reduced_error_pruning(X_val, y_val)

    path = list(c45.traverse_c45(clsfr))
    length = len(path)
    assert length == 23
