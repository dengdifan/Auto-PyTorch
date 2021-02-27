import warnings
from typing import Any, Dict, List, Optional, Tuple

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import numpy as np

from sklearn.base import RegressorMixin

from autoPyTorch.constants import STRING_TO_TASK_TYPES
from autoPyTorch.pipeline.base_pipeline import BasePipeline
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.scaling.base_scaler_choice import \
    ScalerChoice
from autoPyTorch.pipeline.components.preprocessing.time_series_preprocessing.TimeSeriesTransformer import \
    TimeSeriesTransformer
from autoPyTorch.pipeline.components.setup.early_preprocessor.EarlyPreprocessing import EarlyPreprocessing
from autoPyTorch.pipeline.components.setup.lr_scheduler.base_scheduler_choice import SchedulerChoice
from autoPyTorch.pipeline.components.setup.network.base_network import NetworkComponent
from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone_choice import NetworkBackboneChoice
from autoPyTorch.pipeline.components.setup.network_head.base_network_head_choice import NetworkHeadChoice
from autoPyTorch.pipeline.components.setup.network_initializer.base_network_init_choice import (
    NetworkInitializerChoice
)
from autoPyTorch.pipeline.components.setup.optimizer.base_optimizer_choice import OptimizerChoice
from autoPyTorch.pipeline.components.training.data_loader.feature_data_loader import FeatureDataLoader
from autoPyTorch.pipeline.components.training.data_loader.time_series_data_loader import TimeSeriesDataLoader
from autoPyTorch.pipeline.components.training.trainer.base_trainer_choice import (
    TrainerChoice
)
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


class TimeSeriesRegressionPipeline(RegressorMixin, BasePipeline):
    """This class is a proof of concept to integrate AutoPyTorch Components

    It implements a pipeline, which includes as steps:

        ->One preprocessing step
        ->One neural network

    Contrary to the sklearn API it is not possible to enumerate the
    possible parameters in the __init__ function because we only know the
    available regressors at runtime. For this reason the user must
    specifiy the parameters by passing an instance of
    ConfigSpace.configuration_space.Configuration.


    Args:
        config (Configuration)
            The configuration to evaluate.
        random_state (Optional[RandomState): random_state is the random number generator

    Attributes:
    Examples
    """

    def __init__(self,
                 config: Optional[Configuration] = None,
                 steps: Optional[List[Tuple[str, autoPyTorchChoice]]] = None,
                 dataset_properties: Optional[Dict[str, Any]] = None,
                 include: Optional[Dict[str, Any]] = None,
                 exclude: Optional[Dict[str, Any]] = None,
                 random_state: Optional[np.random.RandomState] = None,
                 init_params: Optional[Dict[str, Any]] = None,
                 search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
                 ):
        super().__init__(
            config, steps, dataset_properties, include, exclude,
            random_state, init_params, search_space_updates)

    def score(self, X: np.ndarray, y: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """Scores the fitted estimator on (X, y)

        Args:
            X (np.ndarray): input to the pipeline, from which to guess targets
            batch_size (Optional[int]): batch_size controls whether the pipeline
                will be called on small chunks of the data. Useful when calling the
                predict method on the whole array X results in a MemoryError.
        Returns:
            np.ndarray: coefficient of determination R^2 of the prediction
        """
        from autoPyTorch.pipeline.components.training.metrics.utils import get_metrics, calculate_score
        metrics = get_metrics(self.dataset_properties, ['r2'])
        y_pred = self.predict(X, batch_size=batch_size)
        r2 = calculate_score(y, y_pred, task_type=STRING_TO_TASK_TYPES[self.dataset_properties['task_type']],
                             metrics=metrics)['r2']
        return r2

    def _get_hyperparameter_search_space(self,
                                         dataset_properties: Dict[str, Any],
                                         include: Optional[Dict[str, Any]] = None,
                                         exclude: Optional[Dict[str, Any]] = None,
                                         ) -> ConfigurationSpace:
        """Create the hyperparameter configuration space.

        For the given steps, and the Choices within that steps,
        this procedure returns a configuration space object to
        explore.

        Args:
            include (Optional[Dict[str, Any]]): what hyper-parameter configurations
                to honor when creating the configuration space
            exclude (Optional[Dict[str, Any]]): what hyper-parameter configurations
                to remove from the configuration space
            dataset_properties (Optional[Dict[str, Union[str, int]]]): Characteristics
                of the dataset to guide the pipeline choices of components

        Returns:
            cs (Configuration): The configuration space describing the TimeSeriesRegressionPipeline.
        """
        cs = ConfigurationSpace()

        if dataset_properties is None or not isinstance(dataset_properties, dict):
            if not isinstance(dataset_properties, dict):
                warnings.warn('The given dataset_properties argument contains an illegal value.'
                              'Proceeding with the default value')
            dataset_properties = dict()

        if 'target_type' not in dataset_properties:
            dataset_properties['target_type'] = 'time_series_regression'
        if dataset_properties['target_type'] != 'time_series_regression':
            warnings.warn('Time series regression is being used, however the target_type'
                          'is not given as "time_series_regression". Overriding it.')
            dataset_properties['target_type'] = 'time_series_regression'
        # get the base search space given this
        # dataset properties. Then overwrite with custom
        # regression requirements
        cs = self._get_base_search_space(
            cs=cs, dataset_properties=dataset_properties,
            exclude=exclude, include=include, pipeline=self.steps)

        # Here we add custom code, like this with this
        # is not a valid configuration

        self.configuration_space = cs
        self.dataset_properties = dataset_properties
        return cs

    def _get_pipeline_steps(self, dataset_properties: Optional[Dict[str, Any]]) -> List[Tuple[str, autoPyTorchChoice]]:
        """
        Defines what steps a pipeline should follow.
        The step itself has choices given via autoPyTorchChoice.

        Returns:
            List[Tuple[str, autoPyTorchChoice]]: list of steps sequentially exercised
                by the pipeline.
        """
        steps = []  # type: List[Tuple[str, autoPyTorchChoice]]

        default_dataset_properties = {'target_type': 'time_series_regression'}
        if dataset_properties is not None:
            default_dataset_properties.update(dataset_properties)

        steps.extend([
            ("scaler", ScalerChoice(default_dataset_properties)),
            ("preprocessing", EarlyPreprocessing()),
            ("time_series_transformer", TimeSeriesTransformer()),
            ("network_backbone", NetworkBackboneChoice(default_dataset_properties)),
            ("network_head", NetworkHeadChoice(default_dataset_properties)),
            ("network", NetworkComponent()),
            ("network_init", NetworkInitializerChoice(default_dataset_properties)),
            ("optimizer", OptimizerChoice(default_dataset_properties)),
            ("lr_scheduler", SchedulerChoice(default_dataset_properties)),
            ("data_loader", TimeSeriesDataLoader()),
            ("trainer", TrainerChoice(default_dataset_properties)),
        ])
        return steps

    def _get_estimator_hyperparameter_name(self) -> str:
        """
        Returns the name of the current estimator.

        Returns:
            str: name of the pipeline type
        """
        return "time_series_regressor"