"""C4.5 decision tree.

C4.5 is an algorithm used to construct a decision tree.

This file contains two classes, `C45Classifier` and `_TreeNode`, with only the
former being part of the public API and the latter containing most of C4.5.
"""
from itertools import repeat
from typing import List, Tuple, Union

import numpy as np
from more_itertools import collapse
from scipy import stats

from decision_mining.core.baseclassifier import BaseClassifier


def split_info(attribute: np.ndarray, target: np.ndarray) -> float:
    """Calculate the metric SplitInfo.

    SplitInfo(A, T) = - sum((length of T where v) / (length of T) *\
    log2((length of T where v) / (length of T)))

    Args:
        attribute (np.ndarray): Attribute or column of `X` in training data.
        target (np.ndarray): Target or `y` in training data.

    Returns:
        float: SplitInfo.
    """
    _split_info = 0.
    classes = np.unique(attribute)
    for value in classes:
        pkv = attribute == value
        split = np.count_nonzero(pkv) / target.shape[0]
        _split_info += split * np.log2(split)

    return -_split_info


def gain(attribute: np.ndarray, target: np.ndarray) -> float:
    """Information gain equation, used to calculate the metric information gain.

    Gain(A, T) = Entropy(A) - sum((length of T where v) / \
    (length of T) Entropy(A where v) for v in unique(A)

    Args:
        attribute (np.ndarray): Attribute or column of `X` in training data.
        target (np.ndarray): Target or `y` in training data.

    Returns:
        float: Information Gain.
    """
    classes, counts = np.unique(target, return_counts=True)
    entropy = stats.entropy(counts, base=2.)
    sub_entropy = 0.
    for value in np.unique(attribute):
        mask = attribute == value
        _, pkv = np.unique(target[mask], return_counts=True)
        normalizer = np.count_nonzero(mask) / target.shape[0]
        sub_entropy += normalizer * stats.entropy(pkv, base=2.)

    return entropy - sub_entropy


def gain_ratio(attribute: np.ndarray, target: np.ndarray) -> float:
    """The GainRatio equation, calculates the metric information gain ratio.

    The greater the GainRatio, the greater the information gain.

    0 <= GainRatio <= 1

    GainRatio(A, T) = Gain(A, T) / SplitInfo(A, T)

    Args:
        attribute (np.ndarray): Attribute or column of `X` in training data.
        target (np.ndarray): Target or `y` in training data.

    Returns:
        float: GainRatio(A, T).
    """
    # TODO: Vectorise GainRatio equation: speed+ readability-
    # If there's only one value left in the attribute, the GainRatio is 0.
    if (attribute == attribute[0]).all():
        return 0.
    return gain(attribute, target) / split_info(attribute, target)


def find_threshold(attribute: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """Threshold function as defined in "Improved Use of Continuous Attributes in C4.5" by R. Quinlan.

    Args:
        attribute (np.ndarray): Attribute or column of `X` in training data, continuous values.
        target (np.ndarray): Target or `y` in training data.

    Returns:
        Tuple[float, float]: threshold, gainratio of threshold.
    """
    attr_c = np.sort(attribute)  # Sorted copy of attribute
    thresholds = (attr_c[1:] + attr_c[:-1]) / 2  # All possible thresholds
    best_t = 0.  # Threshold with greatest gain ratio
    # Greatest gain ratio (gain ratio at threshold best_t)
    best_gain_ratio = 0.
    for t in thresholds:
        attr_t = attribute < t
        gain_ratio_at_t = gain_ratio(attr_t, target)
        if gain_ratio_at_t > best_gain_ratio:
            best_gain_ratio = gain_ratio_at_t
            best_t = t

    return best_t, best_gain_ratio


class _TreeNode():
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, cont_mask: np.ndarray = None,
                 parent: '_TreeNode' = None, min_objs: int = 1) -> None:
        """Initialises _TreeNode, recursively applies C4.5 through.

        Not intended to be used standalone, call through C45Classifier instead.

        Args:
            X_train (np.ndarray): Subset of training input samples.
            y_train (np.ndarray): Subset of the target values.
            cont_mask (np.ndarray, optional): Mask array for continuous columns. Defaults to None.
            parent (_TreeNoden, optional): Parent node of tree node object, Defaults to None
            min_objs (int, optional): Minimum amount of objects necessary to split. Defaults to 1.
        """
        # Mask array for indexing continuous columns
        self.cont_mask: np.ndarray = cont_mask
        # Threshold to split on for continuous values
        self.threshold: float = None
        # Gain ratio for each attribute
        self.gain_ratios: np.ndarray = None
        # If this TreeNode is a leaf
        self.is_leaf: bool = None
        # Index of attribute to split on
        self.attribute: int = None
        # Most common class in y_train
        self.mod = None
        # Predicted class
        self.pred_class = None
        # Parent node of node, if top node None
        self.parent = parent
        # visited paramenter for pruning algorithm
        self.visited = False

        classes, count = np.unique(y_train, return_counts=True)
        self.mod = classes[np.argmax(count)]  # get most common class

        # Base case 1
        # All the samples in the list belong to the same class.
        # When this happens, it simply creates a leaf node for the decision
        # tree saying to choose that class.

        # MIN_OBJS feature
        # If there are less than samples than  `min_objs`, set leaf to most common class.
        if (is_leaf := (y_train.shape[0] < min_objs or classes.shape[0] == 1)):
            self.pred_class = self.mod
        else:
            self.attribute = self.find_best_split(X_train, y_train)
            # If there's any positive GainRatio
            if self.gain_ratios.any():
                self.nodes = dict()
                if X_train.ndim > 1:
                    attribute_array = X_train[:, self.attribute]
                else:
                    attribute_array = X_train
                # If splitting on categorical attribute
                if self.threshold is None:
                    for value in np.unique(attribute_array):
                        mask = attribute_array == value
                        self.nodes[value] = _TreeNode(
                            X_train=X_train[mask], y_train=y_train[mask], cont_mask=self.cont_mask,
                            parent=self)
                else:
                    # If splitting on continuous attribute
                    mask = attribute_array < self.threshold
                    self.nodes[True] = _TreeNode(
                        X_train=X_train[mask], y_train=y_train[mask], cont_mask=self.cont_mask,
                        parent=self)
                    self.nodes[False] = _TreeNode(
                        X_train=X_train[~mask], y_train=y_train[~mask], cont_mask=self.cont_mask,
                        parent=self)
            else:
                # Base case 2
                # None of the features provide any information gain.
                # In this case, C4.5 creates a decision node higher up the tree
                # using the expected value of the class.

                # this interpretation of the 2nd base case is based on other
                # implementations, the description of base cases given for ID3
                # and "C4.5: PROGRAMS FOR MACHINE LEARNING" by Ross Quinlan.
                # I am currently not able to verify this interpretation fully.
                is_leaf = True
                self.pred_class = self.mod

        self.is_leaf: bool = is_leaf

    def predict(self, X: np.ndarray) -> Union[str, int]:
        """Predicts the class of X.

        If node is leaf, return pred_class, else go to next node.

        Args:
            X (np.ndarray): Input samples.

        Returns:
            Union[str, int]: Predicted class for `X`.
        """
        if self.is_leaf:
            return self.pred_class
        else:
            value = X[self.attribute]
            if self.threshold is not None:
                value = value < self.threshold
        try:
            next_node: _TreeNode = self.nodes[value]
        except KeyError:
            # Base case 3
            # Instance of previously-unseen class encountered.
            # Again, C4.5 creates a decision node higher up the tree using the
            # expected value.

            # If value isn't in tree, return most common class in subset.
            # this interpretation of the 3rd base case is based on other
            # implementations, the description of base cases given for ID3
            # and "C4.5: PROGRAMS FOR MACHINE LEARNING" by Ross Quinlan.
            # I am currently not able to verify this interpretation fully.
            return self.mod
        else:
            return next_node.predict(X)

    def find_best_split(self, X_train: np.ndarray, y_train: np.ndarray) -> int:
        """Discover best attribute to split on.

        Args:
            X_train (np.ndarray): Subset of training input samples.
            y_train (np.ndarray): Subset of the target values.

        Returns:
            int: Index of column on which to split.
        """
        # Faster solution, needs tinkering
        # ufunc_gain_ratio = np.vectorize(gain_ratio)
        if X_train.ndim > 1:
            gain_ratios = np.zeros(X_train.shape[1], np.float32)

            if self.cont_mask is not None:
                # Set categorical values
                categorical = X_train.T[~self.cont_mask]
                gain_ratios[~self.cont_mask] = np.fromiter(
                    map(gain_ratio, categorical, repeat(y_train)), dtype=np.float32)

                # Set continuous values
                continuous = X_train.T[self.cont_mask]
                thresholds = np.zeros(gain_ratios.shape)
                thresholds[self.cont_mask], gain_ratios[self.cont_mask] = zip(
                    *map(find_threshold, continuous, repeat(y_train)))
            else:
                gain_ratios = np.fromiter(
                    map(gain_ratio, X_train.T, repeat(y_train)), dtype=np.float32)

        else:
            gain_ratios = np.array([gain_ratio(X_train, y_train)])
        self.gain_ratios: np.ndarray = gain_ratios
        split_idx = np.argmax(gain_ratios)
        # If the best split / split index is continuous, also set threshold
        if self.cont_mask is not None and self.cont_mask[split_idx]:
            self.threshold: float = thresholds[split_idx]
        return split_idx


class C45Classifier(BaseClassifier):
    """C4.5 classifier.

    Args:
        continuous_cols (np.ndarray): Index array for continuous columns. Defaults to None.
        min_objs (int, optional): Minimum amount of objects necessary to split. Defaults to 1.
    """

    def __init__(self, continuous_cols: np.ndarray = None, min_objs: int = 1) -> None:
        """Init for C45Classifier.

        Args:
            continuous_cols (np.ndarray): Index array for continuous columns. Defaults to None.
            min_objs (int, optional): Minimum amount of objects necessary to split. Defaults to 1.
        """
        super().__init__()
        self.continuous_cols = continuous_cols
        self.min_objs = min_objs

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Builds a classifier from the training set.

        Args:
            X_train (np.ndarray): Training input samples.
            y_train (np.ndarray): The target values.
        """
        self.y_dtype = y_train.dtype
        self.x_dtype = X_train.dtype
        self.num_attrs = X_train.shape[1] if X_train.ndim == 2 else 1

        if self.continuous_cols is not None:
            mask = np.zeros(self.num_attrs, dtype=bool)
            mask[self.continuous_cols] = True
        else:
            mask = None

        self.node = _TreeNode(X_train=X_train, y_train=y_train,
                              cont_mask=mask, min_objs=self.min_objs)

    def predict(self, X: np.ndarray) -> Union[np.ndarray]:
        """Predict classes for X.

        Args:
            X (np.ndarray): Input samples.

        Returns:
            Union[np.ndarray]: The predicted class(es).
        """
        if not (isinstance(X, np.ndarray) and X.dtype == self.x_dtype):
            errtype = X.dtype if isinstance(X, np.ndarray) else "NaN"
            raise TypeError(
                f"Expected X to be np.ndarray with dtype {self.x_dtype}, got\
 {type(X)} with dtype {errtype} instead.")

        if X.ndim == 2 and X.shape[1] == self.num_attrs:
            return np.fromiter(map(self.node.predict, X), dtype=self.y_dtype)
        elif X.ndim == 1 and X.shape[0] == self.num_attrs:
            return self.node.predict(X)
        else:
            raise ValueError(
                f"Expected X.shape to be (n,{self.num_attrs},), not {X.shape}")

    def reduced_error_pruning(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Prunes the C4.5 Decision Tree based on Quinlan [1987b] suggestion.

        Goes trough all leaves and tests if the parent aka the Branches are needed.
        If a branch turned to leaf performs equal or better then the original_score or \
        best_score achieved till then.
        The branch will be turned to a leaf permanently and its score will be the best_score.

        Args:
            X_val (np.ndarray): Validation input samples.
            y_val (np.ndarray): The target values.
        """
        original_score = self.score(X_val, y_val)
        best_score = original_score
        # Sort path to get the deepest Leafs first.
        path_sorted = sorted(list(traverse_c45(self)), key=lambda x: len(x), reverse=True)
        list_of_leaves = [path[-1][0] for path in path_sorted]
        for leaf in list_of_leaves:
            # go to parent aka the branch
            branch = leaf.parent
            # Skip if branch is already visited
            if not branch or branch.visited:  # pragma: no cover
                continue
            # check if all children of the branch are leaves
            if not all([sub_node.is_leaf for sub_node in branch.nodes.values()]):
                continue
            # Turn Branch into Leaf
            branch.is_leaf = True
            branch.pred_class = branch.mod

            score = self.score(X_val, y_val)
            # If score is worse then the best score achieved
            if score < best_score:
                # revert back to branch
                branch.is_leaf = False
                branch.pred_class = None
            else:
                # if score is beter then previous best_score.
                # turn best_score to the value of score.
                best_score = score
                # add new leaf to list of leaves so its parent can be checked later.
                list_of_leaves.append(branch)


def traverse_node(node: _TreeNode, parents: Tuple[_TreeNode, int, float, bool]) -> List[
        List[Union[Tuple[_TreeNode, int, float, bool], Tuple[_TreeNode, any]]]]:
    """Traverses a c45._TreeNode until leaf node is reached. List depth may be deeper than expected.

    Args:
        node (c45._TreeNode): C4.5 tree node.
        parents (Tuple[c45._TreeNode, int, float, bool]): TreeNode, attribute, threshold, key.

    Returns:
        List[ List[Union[Tuple[c45._TreeNode, int, float, bool], Tuple[c45._TreeNode, any]]]]: \
            C45 decision path, last value is always leaf node. May be many layers deep depending\
                 on tree structure.
    """
    if node.is_leaf:
        return (*parents, (node, node.pred_class))
    return [traverse_node(sub_node, (*parents, (node, node.attribute, node.threshold, key)))
            for key, sub_node in node.nodes.items()]


def traverse_c45(clss: C45Classifier) -> List[List[Union[Tuple[_TreeNode, int, float, bool],
                                                         Tuple[_TreeNode, any]]]]:
    """Traverses a C45Classifier's nodes, returns rules.

    Args:
        clss (c45.C45Classifier): Trained C45Classifier.

    Returns:
        List[ List[Union[Tuple[c45._TreeNode, int, float, bool], Tuple[c45._TreeNode, any]]]]: \
            C45 decision path, last value is always leaf node.
    """
    top_node = clss.node
    return collapse(traverse_node(top_node, tuple()), base_type=tuple)
