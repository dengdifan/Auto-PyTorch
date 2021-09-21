from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Type, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

import pandas as pd

import scipy.sparse

import torch
from torch.utils.data.dataloader import default_collate

HyperparameterValueType = Union[int, str, float]


class FitRequirement(NamedTuple):
    """
    A class that holds inputs required to fit a pipeline. Also indicates whether
    requirements have to be user specified or are generated by the pipeline itself.

    Attributes:
        name (str): The name of the variable expected in the input dictionary
        supported_types (Iterable[Type]): An iterable of all types that are supported
        user_defined (bool): If false, this requirement does not have to be given to the pipeline
        dataset_property (bool): If True, this requirement is automatically inferred
            by the Dataset class
    """

    name: str
    supported_types: Iterable[Type]
    user_defined: bool
    dataset_property: bool

    def __str__(self) -> str:
        """
        String representation for the requirements
        """
        return "Name: %s | Supported types: %s | User defined: %s | Dataset property: %s" % (
            self.name, self.supported_types, self.user_defined, self.dataset_property)


class HyperparameterSearchSpace(NamedTuple):
    """
    A class that holds the search space for an individual hyperparameter.
    Attributes:
        hyperparameter (str):
            name of the hyperparameter
        value_range (Sequence[HyperparameterValueType]):
            range of the hyperparameter, can be defined as min and
            max values for Numerical hyperparameter or a list of
            choices for a Categorical hyperparameter
        default_value (HyperparameterValueType):
            default value of the hyperparameter
        log (bool):
            whether to sample hyperparameter on a log scale
    """
    hyperparameter: str
    value_range: Sequence[HyperparameterValueType]
    default_value: HyperparameterValueType
    log: bool = False

    def __str__(self) -> str:
        """
        String representation for the Search Space
        """
        return "Hyperparameter: %s | Range: %s | Default: %s | log: %s" % (
            self.hyperparameter, self.value_range, self.default_value, self.log)


def custom_collate_fn(batch: List) -> List[Optional[torch.Tensor]]:
    """
    In the case of not providing a y tensor, in a
    dataset of form {X, y}, y would be None.

    This custom collate function allows to yield
    None data for functions that require only features,
    like predict.

    Args:
        batch (List): a batch from a dataset

    Returns:
        List[Optional[torch.Tensor]]
    """

    items = list(zip(*batch))

    # The feature will always be available
    items[0] = default_collate(items[0])
    if None in items[1]:
        items[1] = list(items[1])
    else:
        items[1] = default_collate(items[1])
    return items


def dict_repr(d: Optional[Dict[Any, Any]]) -> str:
    """ Display long message in dict as it is. """
    if isinstance(d, dict):
        return "\n".join(["{}: {}".format(k, v) for k, v in d.items()])
    else:
        return "None"


def replace_string_bool_to_bool(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Utility function to replace string-type bool to
    bool when a dict is read from json

    Args:
        dictionary (Dict[str, Any])
    Returns:
        Dict[str, Any]
    """
    for key, item in dictionary.items():
        if isinstance(item, str):
            if item.lower() == "true":
                dictionary[key] = True
            elif item.lower() == "false":
                dictionary[key] = False
    return dictionary


def get_device_from_fit_dictionary(X: Dict[str, Any]) -> torch.device:
    """
    Get a torch device object by checking if the fit dictionary specifies a device. If not, or if no GPU is available
    return a CPU device.

    Args:
        X (Dict[str, Any]): A fit dictionary to control how the pipeline is fitted
            See autoPyTorch/pipeline/components/base_component.py::autoPyTorchComponent for more details
            about fit_dictionary

    Returns:
        torch.device: Device to be used for training/inference
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device(X.get("device", "cpu"))


def subsampler(data: Union[np.ndarray, pd.DataFrame, scipy.sparse.csr_matrix],
               x: Union[np.ndarray, List[int]]
               ) -> Union[np.ndarray, pd.DataFrame, scipy.sparse.csr_matrix]:
    return data[x] if isinstance(data, (np.ndarray, scipy.sparse.csr_matrix)) else data.iloc[x]


def get_hyperparameter(hyperparameter: HyperparameterSearchSpace,
                       hyperparameter_type: Type[Hyperparameter]) -> Hyperparameter:
    """
    Given a hyperparameter search space, return a ConfigSpace Hyperparameter
    Args:
        hyperparameter (HyperparameterSearchSpace):
            the search space for the hyperparameter
        hyperparameter_type (Hyperparameter):
            the type of the hyperparameter

    Returns:
        Hyperparameter
    """
    if len(hyperparameter.value_range) == 0:
        raise ValueError(hyperparameter.hyperparameter + ': The range has to contain at least one element')
    if len(hyperparameter.value_range) == 1 and hyperparameter_type != CategoricalHyperparameter:
        return Constant(hyperparameter.hyperparameter, hyperparameter.value_range[0])
    if len(hyperparameter.value_range) == 2 and hyperparameter.value_range[0] == hyperparameter.value_range[1]:
        return Constant(hyperparameter.hyperparameter, hyperparameter.value_range[0])
    if hyperparameter_type == CategoricalHyperparameter:
        return CategoricalHyperparameter(hyperparameter.hyperparameter,
                                         choices=hyperparameter.value_range,
                                         default_value=hyperparameter.default_value)
    if hyperparameter_type == UniformFloatHyperparameter:
        assert len(hyperparameter.value_range) == 2, \
            "Float HP range update for %s is specified by the two upper " \
            "and lower values. %s given." % (hyperparameter.hyperparameter, len(hyperparameter.value_range))
        return UniformFloatHyperparameter(hyperparameter.hyperparameter,
                                          lower=hyperparameter.value_range[0],
                                          upper=hyperparameter.value_range[1],
                                          log=hyperparameter.log,
                                          default_value=hyperparameter.default_value)
    if hyperparameter_type == UniformIntegerHyperparameter:
        assert len(hyperparameter.value_range) == 2, \
            "Int HP range update for %s is specified by the two upper " \
            "and lower values. %s given." % (hyperparameter.hyperparameter, len(hyperparameter.value_range))
        return UniformIntegerHyperparameter(hyperparameter.hyperparameter,
                                            lower=hyperparameter.value_range[0],
                                            upper=hyperparameter.value_range[1],
                                            log=hyperparameter.log,
                                            default_value=hyperparameter.default_value)
    raise ValueError('Unknown type: %s for hp %s' % (hyperparameter_type, hyperparameter.hyperparameter))


def add_hyperparameter(cs: ConfigurationSpace,
                       hyperparameter: HyperparameterSearchSpace,
                       hyperparameter_type: Type[Hyperparameter]) -> None:
    """
    Adds the given hyperparameter to the given configuration space

    Args:
        cs (ConfigurationSpace):
            Configuration space where the hyperparameter must be added
        hyperparameter (HyperparameterSearchSpace):
            search space of the hyperparameter
        hyperparameter_type (Hyperparameter):
            type of the hyperparameter

    Returns:
        None
    """
    cs.add_hyperparameter(get_hyperparameter(hyperparameter, hyperparameter_type))
