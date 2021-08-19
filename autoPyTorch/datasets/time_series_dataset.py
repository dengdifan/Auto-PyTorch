from typing import Any, Dict, List, Optional, Tuple, Union, cast
import warnings
import bisect

import numpy as np

import pandas as pd
from scipy.sparse import issparse

from torch.utils.data.dataset import Dataset, Subset, ConcatDataset


import torchvision.transforms

from autoPyTorch.constants import (
    CLASSIFICATION_OUTPUTS,
    CLASSIFICATION_TASKS,
    REGRESSION_OUTPUTS,
    STRING_TO_OUTPUT_TYPES,
    STRING_TO_TASK_TYPES,
    TASK_TYPES_TO_STRING,
    TIMESERIES_CLASSIFICATION,
    TIMESERIES_REGRESSION,
    TIMESERIES_FORECASTING,
)
from autoPyTorch.data.base_validator import BaseInputValidator
from autoPyTorch.datasets.base_dataset import BaseDataset, type_check, type_of_target, TransformSubset
from autoPyTorch.datasets.resampling_strategy import (
    DEFAULT_RESAMPLING_PARAMETERS,
    CrossValTypes,
    HoldoutValTypes,
    get_cross_validators,
    get_holdout_validators,
    is_stratified,
)

from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.data.time_series_forecasting_validator import TimeSeriesForecastingInputValidator
from autoPyTorch.utils.common import FitRequirement, hash_array_or_matrix
from autoPyTorch.datasets.tabular_dataset import TabularDataset

#TIME_SERIES_FORECASTING_INPUT = Tuple[np.ndarray, np.ndarray]  # currently only numpy arrays are supported
#TIME_SERIES_REGRESSION_INPUT = Tuple[np.ndarray, np.ndarray]
#TIME_SERIES_CLASSIFICATION_INPUT = Tuple[np.ndarray, np.ndarray]


class TimeSeriesSequence(BaseDataset):
    def __init__(self,
                 X: Union[np.ndarray, pd.DataFrame],
                 Y: Union[np.ndarray, pd.Series],
                 X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 Y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 dataset_name: Optional[str] = None,
                 resampling_strategy: Union[CrossValTypes, HoldoutValTypes] = HoldoutValTypes.time_series_hold_out_validation,
                 resampling_strategy_args: Optional[Dict[str, Any]] = None,
                 shuffle: bool = False,
                 seed: Optional[int] = 42,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 n_prediction_steps: int = 1,
                 do_split=True,
                 ):
        """
        A dataset representing a time series sequence.
        Args:
            train_tensors:
            dataset_name:
            val_tensors:
            test_tensors:
            resampling_strategy:
            resampling_strategy_args:
            seed:
            train_transforms:
            val_transforms:
            n_prediction_steps: int, how many steps need to be predicted in advance
        """
        train_tensors = (X, Y)
        test_tensors = (X_test, Y_test)
        self.n_prediction_steps = n_prediction_steps

        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = hash_array_or_matrix(train_tensors[0])

        self.train_tensors = train_tensors
        self.val_tensors = None
        self.test_tensors = test_tensors

        self.rand = np.random.RandomState(seed=seed)
        self.shuffle = shuffle

        if do_split:
            self.resampling_strategy = resampling_strategy
            self.resampling_strategy_args = resampling_strategy_args

            # we only allow time series cross validation and holdout validation
            self.cross_validators = get_cross_validators(CrossValTypes.time_series_cross_validation)
            self.holdout_validators = get_holdout_validators(HoldoutValTypes.time_series_hold_out_validation)

        # We also need to be able to transform the data, be it for pre-processing
        # or for augmentation
        self.train_transform = train_transforms
        self.val_transform = val_transforms

    def __getitem__(self, index: int, train: bool = True) -> Tuple[np.ndarray, ...]:
        """
        get a subsequent of time series data, unlike vanilla tabular dataset, we obtain all the previous sequences
        until the given index, this allows us to do further transformation when the

        Args:
            index (int): what element to yield from all the train/test tensors
            train (bool): Whether to apply a train or test transformation, if any

        Returns:
            A transformed single point prediction
        """
        if index < 0 :
            index = self.__len__() + 1 - index

        if hasattr(self.train_tensors[0], 'loc'):
            X = self.train_tensors[0].iloc[:index + 1]
        else:
            X = self.train_tensors[0][:index + 1]

        if self.train_transform is not None and train:
            X = self.train_transform(X)
        elif self.val_transform is not None and not train:
            X = self.val_transform(X)

        # In case of prediction, the targets are not provided
        Y = self.train_tensors[1]
        if Y is not None:
            # Y = Y[:index + self.n_prediction_steps]
            Y = Y[index]
        else:
            Y = None

        return X, Y

    def __len__(self) -> int:
        return self.train_tensors[0].shape[0]

    def get_splits_from_resampling_strategy(self) -> List[Tuple[List[int], List[int]]]:
        """
        Creates a set of splits based on a resampling strategy provided, apart from the
        'get_splits_from_resampling_strategy' implemented in base_dataset, here we will get self.upper_window_size
        with the given value

        Returns
            (List[Tuple[List[int], List[int]]]): splits in the [train_indices, val_indices] format
        """
        splits = []
        if isinstance(self.resampling_strategy, HoldoutValTypes):
            val_share = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                'val_share', None)
            if self.resampling_strategy_args is not None:
                val_share = self.resampling_strategy_args.get('val_share', val_share)
            splits.append(self.create_holdout_val_split(holdout_val_type=self.resampling_strategy,
                                                        val_share=val_share))

            if self.val_tensors is not None:
                upper_window_size = self.__len__() - self.n_prediction_steps
            else:
                upper_window_size = int(self.__len__() * val_share) - self.n_prediction_steps

        elif isinstance(self.resampling_strategy, CrossValTypes):
            num_splits = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                'num_splits', None)
            if self.resampling_strategy_args is not None:
                num_splits = self.resampling_strategy_args.get('num_splits', num_splits)
            # Create the split if it was not created before
            splits.extend(self.create_cross_val_splits(
                    cross_val_type=self.resampling_strategy,
                    num_splits=cast(int, num_splits),
            ))
            upper_window_size = (self.__len__() // num_splits) - self.n_prediction_steps
        else:
            raise ValueError(f"Unsupported resampling strategy={self.resampling_strategy}")
        self.upper_window_size = upper_window_size
        return splits

    def create_cross_val_splits(
        self,
        cross_val_type: CrossValTypes,
        num_splits: int
    ) -> List[Tuple[Union[List[int], np.ndarray], Union[List[int], np.ndarray]]]:
        """
        This function creates the cross validation split for the given task.

        It is done once per dataset to have comparable results among pipelines
        Args:
            cross_val_type (CrossValTypes):
            num_splits (int): number of splits to be created

        Returns:
            (List[Tuple[List[int], List[int]]]): splits in the [train_indices, val_indices] format
        """
        # Create just the split once
        # This is gonna be called multiple times, because the current dataset
        # is being used for multiple pipelines. That is, to be efficient with memory
        # we dump the dataset to memory and read it on a need basis. So this function
        # should be robust against multiple calls, and it does so by remembering the splits
        if not isinstance(cross_val_type, CrossValTypes):
            raise NotImplementedError(f'The selected `cross_val_type` "{cross_val_type}" is not implemented.')
        kwargs = {"n_prediction_steps": self.n_prediction_steps}
        split = self.cross_validators[cross_val_type.name](num_splits, **kwargs)
        return split

    def create_holdout_val_split(
        self,
        holdout_val_type: HoldoutValTypes,
        val_share: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function creates the holdout split for the given task.

        It is done once per dataset to have comparable results among pipelines
        Args:
            holdout_val_type (HoldoutValTypes):
            val_share (float): share of the validation data

        Returns:
            (Tuple[np.ndarray, np.ndarray]): Tuple containing (train_indices, val_indices)
        """
        if holdout_val_type is None:
            raise ValueError(
                '`val_share` specified, but `holdout_val_type` not specified.'
            )
        if self.val_tensors is not None:
            raise ValueError(
                '`val_share` specified, but the Dataset was a given a pre-defined split at initialization already.')
        if val_share < 0 or val_share > 1:
            raise ValueError(f"`val_share` must be between 0 and 1, got {val_share}.")
        if not isinstance(holdout_val_type, HoldoutValTypes):
            raise NotImplementedError(f'The specified `holdout_val_type` "{holdout_val_type}" is not supported.')
        kwargs = {"n_prediction_steps": self.n_prediction_steps}
        train, val = self.holdout_validators[holdout_val_type.name](val_share, self._get_indices(), **kwargs)
        return train, val


class TimeSeriesForecastingDataset(BaseDataset, ConcatDataset):
    datasets: List[TimeSeriesSequence]
    cumulative_sizes: List[int]

    def __init__(self,
                 X: Union[np.ndarray, List[List]],
                 Y: Union[np.ndarray, pd.Series],
                 X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 Y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 dataset_name: Optional[str] = None,
                 resampling_strategy: Union[CrossValTypes, HoldoutValTypes] = HoldoutValTypes.time_series_hold_out_validation,
                 resampling_strategy_args: Optional[Dict[str, Any]] = None,
                 shuffle: Optional[bool] = True,
                 seed: Optional[int] = 42,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 validator: Optional[TimeSeriesForecastingInputValidator] = None,
                 n_prediction_steps: int = 1,
                 shift_input_data: bool = True,
                 normalize_y: bool = True,
                 ):
        """
        :param target_variables: The indices of the variables you want to forecast
        :param sequence_length: The amount of past data you want to use to forecast future value
        :param n_steps: The number of steps you want to forecast into the future
        :param train: Tuple with one tensor holding the training data
        :param val: Tuple with one tensor holding the validation data
        :param shift_input_data: bool
        if the input X and targets needs to be shifted to be aligned:
        such that the data until X[t] is applied to predict the value y[t+n_prediction_steps]
        :param normalize_y: bool
        if y values needs to be normalized with mean 0 and variance 1
        """
        assert X is not Y, "Training and Test data needs to belong two different object!!!"
        self.n_prediction_steps = n_prediction_steps
        if validator is None:
            validator = TimeSeriesForecastingInputValidator(is_classification=False)
        self.validator = validator

        if not isinstance(validator, TimeSeriesForecastingInputValidator):
            raise ValueError(f"This dataset only support TimeSeriesForecastingInputValidator "
                             f"but receive {type(validator)}")

        self.validator.fit(X_train=X, y_train=Y, X_test=X_test, y_test=Y_test,)

        self.numerical_columns = self.validator.feature_validator.numerical_columns
        self.categorical_columns = self.validator.feature_validator.categorical_columns

        self.num_features = self.validator.feature_validator.num_features  # type: int
        self.num_target = self.validator.target_validator.out_dimensionality  # type: int

        X, Y = self.validator.transform(X, Y)

        self.shuffle = shuffle
        self.rand = np.random.RandomState(seed=seed)

        self.resampling_strategy = resampling_strategy
        self.resampling_strategy_args = resampling_strategy_args

        # We also need to be able to transform the data, be it for pre-processing
        # or for augmentation
        self.train_transform = train_transforms
        self.val_transform = val_transforms

        self.num_sequences = len(X)
        self.sequence_lengths = [0] * self.num_sequences
        if shift_input_data:
            for seq_idx in range(self.num_sequences):
                X[seq_idx] = X[seq_idx][:-n_prediction_steps]
                Y[seq_idx] = Y[seq_idx][n_prediction_steps:]
                self.sequence_lengths[seq_idx] = len(X[seq_idx])
        else:
            for seq_idx in range(self.num_sequences):
                self.sequence_lengths[seq_idx] = len(X[seq_idx])

        num_train_data = np.sum(self.sequence_lengths)
        X_train_flatten = np.empty([num_train_data, self.num_features])
        Y_train_flatten = np.empty([num_train_data, self.num_target])
        start_idx = 0

        self.sequences = []
        # flatten the sequences to allow data preprocessing

        for seq_idx, seq_length in enumerate(self.sequence_lengths):
            end_idx = start_idx + seq_length
            X_train_flatten[start_idx: end_idx] = np.array(X[seq_idx]).reshape([-1, self.num_features])
            Y_train_flatten[start_idx: end_idx] = np.array(Y[seq_idx]).reshape([-1, self.num_target])
            start_idx = end_idx

        sequence_lengths_test = [0] * self.num_sequences

        if X_test is not None or Y_test is not None:
            for seq_idx in range(self.num_sequences):
                sequence_lengths_test[seq_idx] = len(X_test[seq_idx])
            num_test_data = np.sum(sequence_lengths_test)
            X_test_flatten = np.empty([num_test_data, self.num_features])
            Y_test_flatten = np.empty([num_test_data, self.num_target])
            start_idx = 0

            for seq_idx, seq_length in enumerate(sequence_lengths_test):
                end_idx = start_idx + seq_length
                X_test_flatten[start_idx: end_idx] = np.array(X_test[seq_idx]).reshape([-1, self.num_features])
                Y_test_flatten[start_idx: end_idx] = np.array(Y_test[seq_idx]).reshape([-1, self.num_target])
                start_idx = end_idx

        if dataset_name is None:
            self.dataset_name = hash_array_or_matrix(X_train_flatten)
        else:
            self.dataset_name = dataset_name
        dataset_name_seqs = [f"{dataset_name}_sequence_{i}" for i in range(self.num_sequences)]

        if normalize_y:
            self.y_train_mean = np.mean(Y_train_flatten)
            self.y_train_std = np.std(Y_train_flatten)
            Y_train_flatten = (Y_train_flatten - self.y_train_mean) / self.y_train_std
            if Y_test is not None:
                Y_test_flatten = (Y_test_flatten - self.y_train_mean) / self.y_train_std
        else:
            self.y_train_mean = 0
            self.y_train_std = 1

        # initialize datasets
        sequences_kwargs = {"resampling_strategy": resampling_strategy,
                            "resampling_strategy_args": resampling_strategy_args,
                            "train_transforms": self.train_transform,
                            "val_transforms": self.val_transform,
                            "n_prediction_steps": n_prediction_steps}
        idx_start_train = 0
        idx_start_test = 0
        sequence_datasets = []


        if X_test is None or Y_test is None:
            for seq_idx, seq_length_train in enumerate(self.sequence_lengths):
                idx_end_train = idx_start_train + seq_length_train
                sequence = TimeSeriesSequence(X=X_train_flatten[idx_start_train: idx_end_train],
                                              Y=Y_train_flatten[idx_start_train: idx_end_train],
                                              dataset_name=dataset_name_seqs[seq_idx],
                                              seed=self.rand.randint(0, 2**20),
                                              **sequences_kwargs)
                sequence_datasets.append(sequence)
                idx_start_train = idx_end_train

                self.sequence_lengths[seq_idx] = len(sequence)
        else:
            for seq_idx, (seq_length_train, seq_length_test) in enumerate(zip(self.sequence_lengths, sequence_lengths_test)):
                idx_end_train = idx_start_train + seq_length_train
                idx_end_test = idx_start_test + seq_length_test
                sequence = TimeSeriesSequence(X=X_train_flatten[idx_start_train: idx_end_train],
                                              Y=Y_train_flatten[idx_start_train: idx_end_train],
                                              X_test=X_test_flatten[idx_start_test: idx_end_test],
                                              Y_test=Y_test_flatten[idx_start_test: idx_end_test],
                                              dataset_name=dataset_name_seqs[seq_idx],
                                              seed=self.rand.randint(0, 2**20),
                                              **sequences_kwargs)
                sequence_datasets.append(sequence)
                idx_start_train = idx_end_train

                self.sequence_lengths[seq_idx] = len(sequence)

        ConcatDataset.__init__(self, datasets=sequence_datasets)

        self.train_tensors = (X_train_flatten, Y_train_flatten)
        if X_test is not None or Y_test is not None:
            self.test_tensors = (X_test_flatten, Y_test_flatten)
        else:
            self.test_tensors = None
        self.val_tensors = None

        self.task_type: Optional[str] = None
        self.issparse: bool = issparse(self.train_tensors[0])
        # TODO find a way to edit input shape!
        self.input_shape: Tuple[int] = (np.min(self.sequence_lengths),self.num_features)

        if len(self.train_tensors) == 2 and self.train_tensors[1] is not None:
            self.output_type: str = type_of_target(self.train_tensors[1])

            if self.output_type in ["binary", "multiclass"]:
                self.output_type = "continuous"

            if STRING_TO_OUTPUT_TYPES[self.output_type] in CLASSIFICATION_OUTPUTS:
                self.output_shape = len(np.unique(self.train_tensors[1]))
            else:
                # self.output_shape = self.train_tensors[1].shape[-1] if self.train_tensors[1].ndim > 1 else 1
                self.output_shape = self.train_tensors[1].shape[-1] if self.train_tensors[1].ndim > 1 else 1

        # TODO: Look for a criteria to define small enough to preprocess
        self.is_small_preprocess = False

        self.task_type = TASK_TYPES_TO_STRING[TIMESERIES_FORECASTING]

        self.numerical_features: List[int] = list(range(self.num_features))
        self.categorical_features: List[int] = []

        self.cross_validators = get_cross_validators(CrossValTypes.time_series_cross_validation)
        self.holdout_validators = get_holdout_validators(HoldoutValTypes.time_series_hold_out_validation)

        self.splits = self.get_splits_from_resampling_strategy()


    def __getitem__(self, idx, train=True):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].__getitem__(sample_idx, train)

    def update_transform(self, transform: Optional[torchvision.transforms.Compose],
                         train: bool = True,
                         ) -> 'BaseDataset':
        """
        During the pipeline execution, the pipeline object might propose transformations
        as a product of the current pipeline configuration being tested.

        This utility allows to return a self with the updated transformation, so that
        a dataloader can yield this dataset with the desired transformations

        Args:
            transform (torchvision.transforms.Compose): The transformations proposed
                by the current pipeline
            train (bool): Whether to update the train or validation transform

        Returns:
            self: A copy of the update pipeline
        """
        if train:
            self.train_transform = transform
        else:
            self.val_transform = transform
        for seq in self.datasets:
            seq = seq.update_transform(transform, train)
        return self

    def get_splits_from_resampling_strategy(self) -> List[Tuple[List[int], List[int]]]:
        """
        Creates a set of splits based on a resampling strategy provided, apart from the
        'get_splits_from_resampling_strategy' implemented in base_dataset, here we will get self.upper_sequence_length
        with the given value

        Returns
            (List[Tuple[List[int], List[int]]]): splits in the [train_indices, val_indices] format
        """
        splits = []
        if isinstance(self.resampling_strategy, HoldoutValTypes):
            val_share = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                'val_share', None)
            if self.resampling_strategy_args is not None:
                val_share = self.resampling_strategy_args.get('val_share', val_share)
            splits.append(self.create_holdout_val_split(holdout_val_type=self.resampling_strategy,
                                                        val_share=val_share))

            if self.val_tensors is not None:
                upper_window_size = np.min(self.sequence_lengths) - self.n_prediction_steps
            else:
                upper_window_size = int(np.min(self.sequence_lengths) * 1 - val_share) - self.n_prediction_steps

        elif isinstance(self.resampling_strategy, CrossValTypes):
            num_splits = DEFAULT_RESAMPLING_PARAMETERS[self.resampling_strategy].get(
                'num_splits', None)
            if self.resampling_strategy_args is not None:
                num_splits = self.resampling_strategy_args.get('num_splits', num_splits)
            # Create the split if it was not created before
            splits.extend(self.create_cross_val_splits(
                    cross_val_type=self.resampling_strategy,
                    num_splits=cast(int, num_splits),
            ))
            upper_window_size = (np.min(self.sequence_lengths) // num_splits) - self.n_prediction_steps
        else:
            raise ValueError(f"Unsupported resampling strategy={self.resampling_strategy}")

        self.upper_window_size = upper_window_size
        return splits

    def get_required_dataset_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing required dataset properties to instantiate a pipeline,
        """
        info = super().get_required_dataset_info()
        info.update({
            'task_type': self.task_type,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'upper_window_size': self.upper_window_size,
        })
        return info

    def get_dataset_properties(self, dataset_requirements: List[FitRequirement]) -> Dict[str, Any]:
        dataset_properties = super().get_dataset_properties(dataset_requirements=dataset_requirements)
        dataset_properties.update({'upper_window_size': self.upper_window_size})
        return dataset_properties

    def create_cross_val_splits(
        self,
        cross_val_type: CrossValTypes,
        num_splits: int
    ) -> List[Tuple[Union[List[int], np.ndarray], Union[List[int], np.ndarray]]]:
        """
        This function creates the cross validation split for the given task.

        It is done once per dataset to have comparable results among pipelines
        Args:
            cross_val_type (CrossValTypes):
            num_splits (int): number of splits to be created

        Returns:
            (List[Tuple[Union[List[int], np.ndarray], Union[List[int], np.ndarray]]]):
                list containing 'num_splits' splits.
        """
        # Create just the split once
        # This is gonna be called multiple times, because the current dataset
        # is being used for multiple pipelines. That is, to be efficient with memory
        # we dump the dataset to memory and read it on a need basis. So this function
        # should be robust against multiple calls, and it does so by remembering the splits

        if not isinstance(cross_val_type, CrossValTypes):
            raise NotImplementedError(f'The selected `cross_val_type` "{cross_val_type}" is not implemented.')
        idx_start = 0
        splits = [[[] for _ in range(len(self.datasets))] for _ in range(num_splits)]
        for idx_seq, dataset in enumerate(self.datasets):
            split = dataset.create_cross_val_splits(cross_val_type, num_splits=num_splits)
            for idx_split in range(num_splits):
                splits[idx_split][idx_seq] = idx_start + split[idx_split]
            idx_start += self.sequence_lengths[idx_seq]
        # in this case, splits is stored as :
        #  [ first split, second_split ...]
        #  first_split = [([0], [1]), ([2], [3])] ....
        splits_merged = []
        for i in range(num_splits):
            split = splits[i]
            train_indices = np.hstack([sp[0] for sp in split])
            test_indices = np.hstack([sp[1] for sp in split])
            splits_merged.append((train_indices, test_indices))
        return splits_merged

    def create_holdout_val_split(
        self,
        holdout_val_type: HoldoutValTypes,
        val_share: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function creates the holdout split for the given task.

        It is done once per dataset to have comparable results among pipelines
        Args:
            holdout_val_type (HoldoutValTypes):
            val_share (float): share of the validation data

        Returns:
            (Tuple[np.ndarray, np.ndarray]): Tuple containing (train_indices, val_indices)
        """
        if holdout_val_type is None:
            raise ValueError(
                '`val_share` specified, but `holdout_val_type` not specified.'
            )

        if val_share < 0 or val_share > 1:
            raise ValueError(f"`val_share` must be between 0 and 1, got {val_share}.")
        if not isinstance(holdout_val_type, HoldoutValTypes):
            raise NotImplementedError(f'The specified `holdout_val_type` "{holdout_val_type}" is not supported.')

        splits = [[() for _ in range(len(self.datasets))] for _ in range(2)]
        idx_start = 0
        for idx_seq, dataset in enumerate(self.datasets):
            split = dataset.create_holdout_val_split(holdout_val_type, val_share)
            for idx_split in range(2):
                splits[idx_split][idx_seq] = idx_start + split[idx_split]
            idx_start += self.sequence_lengths[idx_seq]

        train_indices = np.hstack([sp for sp in splits[0]])
        test_indices = np.hstack([sp for sp in splits[1]])

        return train_indices, test_indices


def _check_time_series_forecasting_inputs(train: np.ndarray,
                                          val: Optional[np.ndarray] = None) -> None:
    if train.ndim != 3 or any(isinstance(i, (list, np.ndarray)) for i in train):
        raise ValueError(
            "The training data for time series forecasting has to be a three-dimensional tensor of shape PxLxM. or a"
            "nested list")
    if val is not None:
        if val.ndim != 3 or any(isinstance(i, (list, np.ndarray)) for i in val):
            raise ValueError(
                "The validation data for time series forecasting "
                "has to be a three-dimensional tensor of shape PxLxM or a nested list.")


class TimeSeriesDataset(BaseDataset):
    """
    Common dataset for time series classification and regression data
    Args:
        X (np.ndarray): input training data.
        Y (Union[np.ndarray, pd.Series]): training data targets.
        X_test (Optional[np.ndarray]):  input testing data.
        Y_test (Optional[Union[np.ndarray, pd.DataFrame]]): testing data targets
        resampling_strategy (Union[CrossValTypes, HoldoutValTypes]),
            (default=HoldoutValTypes.holdout_validation):
            strategy to split the training data.
        resampling_strategy_args (Optional[Dict[str, Any]]): arguments
            required for the chosen resampling strategy. If None, uses
            the default values provided in DEFAULT_RESAMPLING_PARAMETERS
            in ```datasets/resampling_strategy.py```.
        shuffle:  Whether to shuffle the data before performing splits
        seed (int), (default=1): seed to be used for reproducibility.
        train_transforms (Optional[torchvision.transforms.Compose]):
            Additional Transforms to be applied to the training data.
        val_transforms (Optional[torchvision.transforms.Compose]):
            Additional Transforms to be applied to the validation/test data.

        Notes: Support for Numpy Arrays is missing Strings.

        """

    def __init__(self,
                 X: np.ndarray,
                 Y: Union[np.ndarray, pd.Series],
                 X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 Y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 resampling_strategy: Union[CrossValTypes, HoldoutValTypes] = HoldoutValTypes.holdout_validation,
                 resampling_strategy_args: Optional[Dict[str, Any]] = None,
                 shuffle: Optional[bool] = True,
                 seed: Optional[int] = 42,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 dataset_name: Optional[str] = None,
                 validator: Optional[BaseInputValidator] = None,
                 ):
        # Take information from the validator, which guarantees clean data for the
        # dataset.
        # TODO: Consider moving the validator to the pipeline itself when we
        # move to using the fit_params on scikit learn 0.24
        if validator is None:
            raise ValueError("A feature validator is required to build a time series pipeline")

        self.validator = validator

        X, Y = self.validator.transform(X, Y)
        if X_test is not None:
            X_test, Y_test = self.validator.transform(X_test, Y_test)

        super().__init__(train_tensors=(X, Y),
                         test_tensors=(X_test, Y_test),
                         shuffle=shuffle,
                         resampling_strategy=resampling_strategy,
                         resampling_strategy_args=resampling_strategy_args,
                         seed=seed, train_transforms=train_transforms,
                         dataset_name=dataset_name,
                         val_transforms=val_transforms)

        if self.output_type is not None:
            if STRING_TO_OUTPUT_TYPES[self.output_type] in CLASSIFICATION_OUTPUTS:
                self.task_type = TASK_TYPES_TO_STRING[TIMESERIES_CLASSIFICATION]
            elif STRING_TO_OUTPUT_TYPES[self.output_type] in REGRESSION_OUTPUTS:
                self.task_type = TASK_TYPES_TO_STRING[TIMESERIES_REGRESSION]
            else:
                raise ValueError(f"Output type {self.output_type} currently not supported ")
        else:
            raise ValueError("Task type not currently supported ")
        if STRING_TO_TASK_TYPES[self.task_type] in CLASSIFICATION_TASKS:
            self.num_classes: int = len(np.unique(self.train_tensors[1]))

        # filter the default cross and holdout validators if we have a regression task
        # since we cannot use stratification there
        if self.task_type == TASK_TYPES_TO_STRING[TIMESERIES_REGRESSION]:
            self.cross_validators = {cv_type: cv for cv_type, cv in self.cross_validators.items()
                                     if not is_stratified(cv_type)}
            self.holdout_validators = {hv_type: hv for hv_type, hv in self.holdout_validators.items()
                                       if not is_stratified(hv_type)}

        self.num_features = self.train_tensors[0].shape[2]
        self.numerical_features: List[int] = list(range(self.num_features))
        self.categorical_features: List[int] = []

    def get_required_dataset_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing required dataset properties to instantiate a pipeline,
        """
        info = super().get_required_dataset_info()
        info.update({
            'task_type': self.task_type,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,

        })
        return info
