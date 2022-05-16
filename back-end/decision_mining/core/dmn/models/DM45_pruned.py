"""Decision Model for the pruned version of the C4.5 algortime."""
import numpy as np
from sklearn.model_selection import train_test_split

from decision_mining.core.dmn.models.DM45 import DM45
from decision_mining.core.c45 import C45Classifier


class DM45_pruned(DM45):
    """Decision Model for the C45 Algortime."""

    def __init__(self) -> None:
        """Initialize the Decision Model."""
        super().__init__()
        model_info = {
            "name": "DM45 Pruned",
            "id": "DM45_pruned",
            "description": (
                "DM45 Pruned is a decision tree model based on the C4.5 algoritme invented by R. "
                "Quinlan and implemented by students of the HU University of Applied Sciences "
                "Utrecht. The model is pruned with reduced error pruning."
                "The model represents the decision process as a tree with branches. "
                "Based on certain conditions found in the data, it branches the tree. "
                "This process repeats itself until it has converged on the most specific rules "
                "possible."),
        }
        parameters = {
            "train_and_prune_split": {
                "value": 0.7,
                "min": 0.01,
                "max": 1,
                "step": 0.05,
                "type": "float",
                "description": "Split percentage for training and pruning."
            }
        }
        self.model_info.update(model_info)
        self.model_info["parameters"].update(parameters)

    def make_model(self, X: np.ndarray, y: np.ndarray,
                   continuous_cols: np.ndarray = None) -> C45Classifier:
        """Make a classifier Model, and train it on X and y.

        Args:
            X (np.ndarray): Training input samples.
            y (np.ndarray): The target values.
            cols (List[str]): columns
            continuous_cols (np.ndarray, optional): Continuous column indices. Defaults to None.

        Returns:
            Trained and pruned classifier.
        """
        X_train, X_prun, y_train, y_prun = train_test_split(
            X, y, train_size=self.model_info["parameters"]["train_and_prune_split"]["value"])
        clsfr = DM45.make_model(self, X_train, y_train, continuous_cols)
        clsfr.reduced_error_pruning(X_prun, y_prun)
        self.trained_model = clsfr
        return clsfr
