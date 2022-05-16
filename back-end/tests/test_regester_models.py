"""Tests for regester_models.py."""
import pytest

from decision_mining import regester_models as r_models
from decision_mining.core.dmn.models.DM45 import DM45


@pytest.fixture
def base_regestry() -> r_models.DecisionModelRegestry:
    """Fixture for DecisionModelRegestry."""
    regestry = r_models.DecisionModelRegestry()
    regestry.register(DM45())
    return regestry


def test_get_model(base_regestry: r_models.DecisionModelRegestry) -> None:
    """Tests get_model function.

    Args:
        base_regestry (r_models.DecisionModelRegestry): model regestry
    """
    with pytest.raises(ValueError):
        base_regestry.get_model("Test")

    assert isinstance(DM45(), type(base_regestry.get_model("DM45"))), \
        "base_regestry.get_model('DM45') is not an instance of DM45"


def test_get_all_models(base_regestry: r_models.DecisionModelRegestry) -> None:
    """Tests get_all_models function.

    Args:
        base_regestry (r_models.DecisionModelRegestry): model regestry
    """
    assert isinstance(DM45(), type(base_regestry.get_all_models()[0])), \
        "base_regestry.get_all_models()[0] is not an instance of DM45"


def test_get_all_models_parameters(base_regestry: r_models.DecisionModelRegestry) -> None:
    """Tests get_all_models_parameters function.

    Args:
        base_regestry (r_models.DecisionModelRegestry): model regestry
    """
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
    assert base_regestry.get_all_models_parameters()["DM45"] == parameters, \
        f"base_regestry.get_all_models_parameters()['DM45'] is not {parameters}"


def test_get_all_models_info(base_regestry: r_models.DecisionModelRegestry) -> None:
    """Tests get_all_models_info function.

    Args:
        base_regestry (r_models.DecisionModelRegestry): model regestry
    """
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
    model_info = {
        "name": "DM45",
        "id": "DM45",
        "description": (
            "DM45 is a decision tree model based on the C4.5 algoritme invented by R. Quinlan and "
            "implemented by students of the HU University of Applied Sciences Utrecht. "
            "The model represents the decision process as a tree with branches. "
            "Based on certain conditions found in the data, it branches the tree. "
            "This process repeats itself until it has converged on the most specific rules "
            "possible."),
        "parameters": parameters
    }
    assert base_regestry.get_all_models_info()["DM45"] == model_info, \
        f"base_regestry.get_all_models_info()['DM45'] is not equal to {model_info}"
