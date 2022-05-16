"""Decision Model for the C4.5 algortime."""
from itertools import repeat
from typing import Callable, List, Tuple, Union

import numpy as np

from decision_mining.core.dmn.models.decisionModel import DecisionModel
from decision_mining.core.c45 import C45Classifier, _TreeNode, traverse_c45
from decision_mining.core.dmn.rule import Rule


class DM45(DecisionModel):
    """Decision Model for the C45 Algortime."""

    def __init__(self) -> None:
        """Initialize the DecisionModel."""
        parameters = {
            "min_objs": {
                "value": 1,
                "min": 1,
                "max": 50,
                "step": 1,
                "type": "int",
                "description": "Minimum number of objects in a node. To consider a rule."
            }
        }
        self.model_info = {
            "name": "DM45",
            "id": "DM45",
            "description": (
                "DM45 is a decision tree model based on the C4.5 algoritme invented by R. Quinlan "
                "and implemented by students of the HU University of Applied Sciences Utrecht. "
                "The model represents the decision process as a tree with branches. "
                "Based on certain conditions found in the data, it branches the tree. "
                "This process repeats itself until it has converged on the most specific rules "
                "possible."),
            "parameters": parameters
        }

    def make_model(self, X: np.ndarray, y: np.ndarray,
                   continuous_cols: np.ndarray = None) -> C45Classifier:
        """Make a classifier Model, and train it on X and y.

        Args:
            X (np.ndarray): Training input samples.
            y (np.ndarray): The target values.
            continuous_cols (np.ndarray, optional): Continuous column indices. Defaults to None.

        Returns:
            Trained classifier.
        """
        clsfr = C45Classifier(continuous_cols=continuous_cols,
                              min_objs=self.model_info["parameters"]["min_objs"]["value"])
        clsfr.fit(X, y)

        return clsfr

    def extract_rules(self, cols: List[str], model: C45Classifier) -> List[Rule]:
        """Extract rules from the classifier model.

        Args:
            cols (List[str]): columns
            model (C45Classifier): Trained classifier.

        Raises:
            ValueError: When the model is not trained.

        Returns:
            List[Rule]: rules
        """
        path = traverse_c45(model)
        return self.make_rules(range(len(cols) - 1), path)

    @staticmethod
    def make_rule(RuleFactory: Callable, path: Union[Tuple[_TreeNode, int, float, bool],
                                                     Tuple[_TreeNode, any]]) -> Rule:
        """Make a Rule object based on a C45Classifier path.

        Args:
            RuleFactory (Callable): Function/Class for creating a Rule object.
            path (Union[Tuple[_TreeNode, int, float, bool],\
            Tuple[_TreeNode, any]]): C45Classifier path.

        Returns:
            Rule: Complete Rule object.
        """
        rule = RuleFactory()
        # For each node in path
        for _, attr, threshold, key in path[:-1]:
            # If the column is continuous, threshold is not None
            if threshold is not None:
                entry = {"threshold": threshold, "<": key}
                if rule.cols[attr] is None:
                    rule.cols[attr] = [entry]
                else:
                    rule.cols[attr].append(entry)
            else:
                # Categorical column can only come up once, set to key
                rule.cols[attr] = key

        # Set decision
        rule.decision = path[-1][1]

        return rule

    @staticmethod
    def make_rules(cols: List[int],
                   path_list: List[Tuple[_TreeNode, int, float, bool]]) -> List[Rule]:
        """Make a list of Rule objects based on a list of C45Classifier decision paths.

        Args:
            cols (list): List of column indices.
            path_list (List[Tuple[_TreeNode, int, float, bool]]): C45Classifier decision paths.

        Returns:
            List[Rule]: List of complete Rule objects.
        """
        RuleFactory = Rule.rule_generator(cols)
        return list(map(DM45.make_rule, repeat(RuleFactory), path_list))
