import torch

import copy
import unittest
from autoPyTorch.pipeline.components.setup.forecasting_target_scaling import TargetScalerChoice
from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.base_target_scaler import BaseTargetScaler
from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.TargetNoScaler import TargetNoScaler
from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.TargetMaxAbsScaler import TargetMaxAbsScaler
from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.TargetMeanAbsScaler import TargetMeanAbsScaler
from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.TargetMinMaxScaler import TargetMinMaxScaler
from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.TargetStandardScaler import TargetStandardScaler


class TestTargetScalar(unittest.TestCase):
    def test_get_set_config_space(self):
        """Make sure that we can setup a valid choice in the encoder
        choice"""
        rescaler_choice = TargetScalerChoice({})
        cs = rescaler_choice.get_hyperparameter_search_space()

        # Make sure that all hyperparameters are part of the search space
        self.assertListEqual(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(list(rescaler_choice.get_components().keys()))
        )

        # Make sure we can properly set some random configs
        # Whereas just one iteration will make sure the algorithm works,
        # doing five iterations increase the confidence. We will be able to
        # catch component specific crashes
        for i in range(5):
            config = cs.sample_configuration()
            config_dict = copy.deepcopy(config.get_dictionary())
            rescaler_choice.set_hyperparameters(config)

            self.assertEqual(rescaler_choice.choice.__class__,
                             rescaler_choice.get_components()[config_dict['__choice__']])

            # Then check the choice configuration
            selected_choice = config_dict.pop('__choice__', None)
            for key, value in config_dict.items():
                # Remove the selected_choice string from the parameter
                # so we can query in the object for it
                key = key.replace(selected_choice + ':', '')
                self.assertIn(key, vars(rescaler_choice.choice))
                self.assertEqual(value, rescaler_choice.choice.__dict__[key])

        include = ['TargetMeanAbsScaler', 'TargetMaxAbsScaler']
        cs = rescaler_choice.get_hyperparameter_search_space(include=include)
        self.assertTrue(
            sorted(cs.get_hyperparameter('__choice__').choices),
            sorted(include),
        )

    def test_target_no_scalar(self):
        X = {'dataset_properties': {}}
        scalar = TargetNoScaler()
        scalar = scalar.fit(X)
        X = scalar.transform(X)
        self.assertIsInstance(X['target_scaler'], BaseTargetScaler)

        past_targets = torch.rand([5, 6, 7])
        future_targets = torch.rand(([5, 3, 7]))

        transformed_past_target, transformed_future_targets, loc, scale = scalar(past_targets,
                                                                                 future_targets=future_targets)
        self.assertTrue(torch.equal(past_targets, transformed_past_target))
        self.assertTrue(torch.equal(future_targets, transformed_future_targets))
        self.assertIsNone(loc)
        self.assertIsNone(scale)

        _, transformed_future_targets, _, _ = scalar(past_targets)
        self.assertIsNone(transformed_future_targets)

    def test_target_max_abs_scalar(self):
        X = {'dataset_properties': {}}
        scalar = TargetMaxAbsScaler()
        scalar = scalar.fit(X)
        X = scalar.transform(X)
        self.assertIsInstance(X['target_scaler'], BaseTargetScaler)

        past_targets = torch.vstack(
            [
                torch.zeros(10),
                torch.Tensor([0.] * 2 + [2.] * 8),
                torch.ones(10) * 4
            ]
        ).unsqueeze(-1)
        past_observed_values = torch.vstack(
            [
                torch.Tensor([False] * 3 + [True] * 7),
                torch.Tensor([False] * 2 + [True] * 8),
                torch.Tensor([True] * 10)

            ]).unsqueeze(-1).bool()
        future_targets = torch.ones([3, 10, 1]) * 10

        transformed_past_target, transformed_future_targets, loc, scale = scalar(
            past_targets, past_observed_values=past_observed_values, future_targets=future_targets
        )

        self.assertTrue(torch.equal(transformed_past_target[0], torch.zeros([10, 1])))
        self.assertTrue(torch.equal(transformed_past_target[1], torch.Tensor([0.] * 2 + [1.] * 8).unsqueeze(-1)))
        self.assertTrue(torch.equal(transformed_past_target[2], torch.ones([10, 1])))

        self.assertTrue(torch.equal(transformed_future_targets[0], torch.ones([10, 1]) * 10))
        self.assertTrue(torch.equal(transformed_future_targets[1], torch.ones([10, 1]) * 5))
        self.assertTrue(torch.equal(transformed_future_targets[2], torch.ones([10, 1]) * 2.5))

        self.assertTrue(
            torch.equal(scale, torch.Tensor([1., 2., 4.]).reshape([len(past_targets), 1, past_targets.shape[-1]]))
        )

        transformed_past_target, transformed_future_targets, loc, scale = scalar(past_targets,
                                                                                 future_targets=future_targets)
        self.assertTrue(torch.equal(transformed_past_target[0], torch.zeros([10, 1])))
        self.assertTrue(torch.equal(transformed_past_target[1], torch.Tensor([0.] * 2 + [1.] * 8).unsqueeze(-1)))
        self.assertTrue(torch.equal(transformed_past_target[2], torch.ones([10, 1])))

        self.assertTrue(torch.equal(transformed_future_targets[0], torch.ones([10, 1]) * 10))
        self.assertTrue(torch.equal(transformed_future_targets[1], torch.ones([10, 1]) * 5))
        self.assertTrue(torch.equal(transformed_future_targets[2], torch.ones([10, 1]) * 2.5))

        self.assertTrue(
            torch.equal(scale, torch.Tensor([1., 2., 4.]).reshape([len(past_targets), 1, past_targets.shape[-1]]))
        )
        self.assertIsNone(loc)

        transformed_past_target_full, transformed_future_targets_full, loc_full, scale_full = scalar(
            past_targets, past_observed_values=torch.ones([2, 10, 1], dtype=torch.bool), future_targets=future_targets
        )
        self.assertTrue(torch.equal(transformed_past_target, transformed_past_target_full))
        self.assertTrue(torch.equal(transformed_future_targets_full, transformed_future_targets_full))
        self.assertTrue(torch.equal(scale, scale_full))

        self.assertIsNone(loc_full)
