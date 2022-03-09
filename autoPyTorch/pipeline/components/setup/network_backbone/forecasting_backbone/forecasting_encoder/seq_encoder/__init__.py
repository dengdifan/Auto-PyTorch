import os
from collections import OrderedDict
from typing import Dict, Optional, List, Any, Union
import numpy as np
from sklearn.pipeline import Pipeline

from ConfigSpace.hyperparameters import (
    Constant,
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    OrdinalHyperparameter,
)
from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from ConfigSpace.conditions import (
    EqualsCondition, OrConjunction, GreaterThanCondition, NotEqualsCondition, AndConjunction
)
from ConfigSpace.forbidden import ForbiddenInClause, ForbiddenEqualsClause, ForbiddenAndConjunction

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType

from autoPyTorch.pipeline.components.base_component import (
    ThirdPartyComponents,
    autoPyTorchComponent,
    find_components,
)
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder import \
    AbstractForecastingEncoderChoice

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder. \
    base_forecasting_encoder import BaseForecastingEncoder
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import \
    ForecastingNetworkStructure
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.other_components.TemporalFusion import \
    TemporalFusion

directory = os.path.split(__file__)[0]
_encoders = find_components(__package__,
                            directory,
                            BaseForecastingEncoder)
_addons = ThirdPartyComponents(BaseForecastingEncoder)


def add_encoder(encoder: BaseForecastingEncoder) -> None:
    _addons.add_component(encoder)


class SeqForecastingEncoderChoice(AbstractForecastingEncoderChoice):
    deepAR_decoder_name = 'MLPDecoder'
    deepAR_decoder_prefix = 'block_1'
    tf_prefix = "temporal_fusion"

    def get_components(self) -> Dict[str, autoPyTorchComponent]:
        """Returns the available backbone components

        Args:
            None

        Returns:
            Dict[str, autoPyTorchComponent]: all basebackbone components available
                as choices for learning rate scheduling
        """
        components = OrderedDict()
        components.update(_encoders)
        components.update(_addons.components)
        return components

    def get_hyperparameter_search_space(
            self,
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            num_blocks: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="num_blocks",
                                                                              value_range=(1, 1),
                                                                              default_value=1),
            variable_selection: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="variable_selection",
                value_range=(True, False),
                default_value=False
            ),
            share_single_variable_networks: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="share_single_variable_networks",
                value_range=(True, False),
                default_value=False,
            ),
            use_temporal_fusion: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter='use_temporal_fusion',
                value_range=(True, False),
                default_value=False,
            ),
            decoder_auto_regressive: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="decoder_auto_regressive",
                value_range=(True, False),
                default_value=True,
            ),
            skip_connection: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="skip_connection",
                                                                                   value_range=(True, False),
                                                                                   default_value=False),
            skip_connection_type: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="skip_connection_type",
                value_range=("add", "gate_add_norm"),
                default_value="gate_add_norm",
            ),
            grn_use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="grn_use_dropout",
                                                                                   value_range=(True, False),
                                                                                   default_value=True),
            grn_dropout_rate: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='grn_dropout_rate',
                                                                                    value_range=(0.0, 0.8),
                                                                                    default_value=0.1),
            default: Optional[str] = None,
            include: Optional[List[str]] = None,
            exclude: Optional[List[str]] = None,
    ) -> ConfigurationSpace:
        """Returns the configuration space of the current chosen components

        Args:
            dataset_properties (Optional[Dict[str, str]]): Describes the dataset to work on
            num_blocks (HyperparameterSearchSpace): number of encoder-decoder structure blocks
            variable_selection (HyperparameterSearchSpace): if variable selection is applied, if True, then the first
                block will be attached with a variable selection block while the following will be enriched with static
                features.
            share_single_variable_networks( HyperparameterSearchSpace): if single variable networks are shared between
                encoder and decoder
            skip_connection: HyperparameterSearchSpace: if skip connection is applied
            use_temporal_fusion (HyperparameterSearchSpace): if temporal fusion layer is applied
            tf_attention_n_head_log (HyperparameterSearchSpace): log value of tf attention dims
            tf_attention_d_model_log (HyperparameterSearchSpace): log value of tf attention d model
            tf_use_dropout (HyperparameterSearchSpace): if tf uses dropout
            tf_dropout_rate (HyperparameterSearchSpace): dropout rate of tf layer
            skip_connection_type (HyperparameterSearchSpace): skip connection type, it could be directly added or a grn
                network (
                Lim et al, Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting:
                https://arxiv.org/abs/1912.09363) TODO consider hidden size of grn as a new HP
            grn_use_dropout (HyperparameterSearchSpace): if dropout layer is applied to GRN, since variable selection
                network also contains GRN, this parameter also influence variable selection network
            grn_dropout_rate (HyperparameterSearchSpace): dropout rate of GRN, same as above, this variable also
                influence variable selection network
            decoder_auto_regressive: HyperparameterSearchSpace: if decoder is auto_regressive, e.g., if the decoder
                receives the output as its input, this only works for  auto_regressive decoder models
            default (Optional[str]): Default backbone to use
            include: Optional[Dict[str, Any]]: what components to include. It is an exhaustive
                list, and will exclusively use this components.
            exclude: Optional[Dict[str, Any]]: which components to skip

        Returns:
            ConfigurationSpace: the configuration space of the hyper-parameters of the
                 chosen component
        """
        if dataset_properties is None:
            dataset_properties = {}

        static_features_shape = dataset_properties.get("static_features_shape", 0)
        future_feature_shapes = dataset_properties.get("future_feature_shapes", (0,))

        cs = ConfigurationSpace()

        min_num_blocks, max_num_blocks = num_blocks.value_range

        variable_selection = get_hyperparameter(variable_selection, CategoricalHyperparameter)
        share_single_variable_networks = get_hyperparameter(share_single_variable_networks, CategoricalHyperparameter)

        decoder_auto_regressive = get_hyperparameter(decoder_auto_regressive, CategoricalHyperparameter)

        if min_num_blocks == max_num_blocks:
            num_blocks = Constant(num_blocks.hyperparameter, num_blocks.value_range[0])
        else:
            num_blocks = OrdinalHyperparameter(
                num_blocks.hyperparameter,
                sequence=list(range(min_num_blocks, max_num_blocks + 1))
            )

        skip_connection = get_hyperparameter(skip_connection, CategoricalHyperparameter)

        hp_network_structures = [num_blocks, decoder_auto_regressive, variable_selection,
                                 skip_connection]
        cond_skip_connections = []
        if True in skip_connection.choices:
            skip_connection_type = get_hyperparameter(skip_connection_type, CategoricalHyperparameter)
            hp_network_structures.append(skip_connection_type)
            cond_skip_connections.append(EqualsCondition(skip_connection_type, skip_connection, True))
            if 'grn' in skip_connection_type.choices:
                grn_use_dropout = get_hyperparameter(grn_use_dropout, CategoricalHyperparameter)
                hp_network_structures.append(grn_use_dropout)
                if True in variable_selection.choices:
                    cond_skip_connections.append(
                        OrConjunction(EqualsCondition(grn_use_dropout, skip_connection_type, "grn"),
                                      EqualsCondition(grn_dropout_rate, variable_selection, True))
                    )
                else:
                    cond_skip_connections.append(EqualsCondition(grn_use_dropout, skip_connection_type, "grn"))
                if True in grn_use_dropout.choices:
                    grn_dropout_rate = get_hyperparameter(grn_dropout_rate, UniformFloatHyperparameter)
                    hp_network_structures.append(grn_dropout_rate)
                    cond_skip_connections.append(EqualsCondition(grn_dropout_rate, grn_use_dropout, True))
        elif True in variable_selection.choices:
            cond_skip_connections.append(EqualsCondition(grn_dropout_rate, variable_selection, True))

        cs.add_hyperparameters(hp_network_structures)
        if cond_skip_connections:
            cs.add_conditions(cond_skip_connections)

        if static_features_shape + future_feature_shapes[-1] == 0:
            if False in variable_selection.choices and False in decoder_auto_regressive.choices:
                if variable_selection.num_choices == 1 and decoder_auto_regressive.num_choices == 1:
                    raise ValueError("When no future information is available, it is not possible to disable variable"
                                     "selection and enable auto-regressive decoder model")
                cs.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(variable_selection, False),
                    ForbiddenEqualsClause(decoder_auto_regressive, False)
                ))
        if True in variable_selection.choices:
            cs.add_hyperparameter(share_single_variable_networks)
            cs.add_condition(EqualsCondition(share_single_variable_networks, variable_selection, True))

        # Compile a list of legal preprocessors for this problem
        available_encoders = self.get_available_components(
            dataset_properties=dataset_properties,
            include=include, exclude=exclude)

        available_decoders = self.get_available_components(
            dataset_properties=dataset_properties,
            include=None, exclude=None,
            components=self.get_decoder_components())

        if len(available_encoders) == 0:
            raise ValueError("No Encoder found")
        if len(available_decoders) == 0:
            raise ValueError("No Decoder found")

        if default is None:
            defaults = self._defaults_network
            for default_ in defaults:
                if default_ in available_encoders:
                    default = default_
                    break
        updates_choice = self._get_search_space_updates()

        forbiddens_decoder_auto_regressive = []

        if False in decoder_auto_regressive.choices:
            forbidden_decoder_ar = ForbiddenEqualsClause(decoder_auto_regressive, True)
        else:
            forbidden_decoder_ar = None

        for i in range(1, int(max_num_blocks) + 1):
            block_prefix = f'block_{i}:'

            if '__choice__' in updates_choice.keys():
                choice_hyperparameter = updates_choice['__choice__']
                if not set(choice_hyperparameter.value_range).issubset(available_encoders):
                    raise ValueError("Expected given update for {} to have "
                                     "choices in {} got {}".format(self.__class__.__name__,
                                                                   available_encoders,
                                                                   choice_hyperparameter.value_range))
                hp_encoder = CategoricalHyperparameter(block_prefix + '__choice__',
                                                       choice_hyperparameter.value_range,
                                                       default_value=choice_hyperparameter.default_value)
            else:
                hp_encoder = CategoricalHyperparameter(
                    block_prefix + '__choice__',
                    list(available_encoders.keys()),
                    default_value=default
                )
            cs.add_hyperparameter(hp_encoder)
            if i > int(min_num_blocks):
                cs.add_condition(
                    GreaterThanCondition(hp_encoder, num_blocks, i - 1)
                )

            decoder2encoder = {key: [] for key in available_decoders.keys()}
            encoder2decoder = {}
            for encoder_name in hp_encoder.choices:
                updates = self._get_search_space_updates(prefix=block_prefix + encoder_name)
                config_space = available_encoders[encoder_name].get_hyperparameter_search_space(dataset_properties,
                                                                                                # type: ignore
                                                                                                **updates)
                parent_hyperparameter = {'parent': hp_encoder, 'value': encoder_name}
                cs.add_configuration_space(
                    block_prefix + encoder_name,
                    config_space,
                    parent_hyperparameter=parent_hyperparameter
                )

                allowed_decoders = available_encoders[encoder_name].allowed_decoders()
                if len(allowed_decoders) > 1:
                    if 'decoder_type' not in config_space:
                        raise ValueError('When a specific encoder has more than one allowed decoder, its ConfigSpace'
                                         'must contain the hyperparameter "decoder_type" ! Please check your encoder '
                                         'setting!')
                    hp_decoder_choice = config_space.get_hyperparameter('decoder_type').choices
                    if not set(hp_decoder_choice).issubset(allowed_decoders):
                        raise ValueError(
                            'The encoder hyperparameter decoder_type must be a subset of the allowed_decoders')
                    allowed_decoders = hp_decoder_choice
                for decoder_name in allowed_decoders:
                    decoder2encoder[decoder_name].append(encoder_name)
                encoder2decoder[encoder_name] = allowed_decoders

            for decoder_name in available_decoders.keys():
                if not decoder2encoder[decoder_name]:
                    continue
                updates = self._get_search_space_updates(prefix=block_prefix + decoder_name)
                if i == 1 and decoder_name == self.deepAR_decoder_name:
                    # TODO this is only a temporary solution, a fix on ConfigSpace needs to be implemented
                    updates['can_be_auto_regressive'] = True
                if decoder_name == "MLPDecoder" and i < int(max_num_blocks):
                    updates['has_local_layer'] = HyperparameterSearchSpace('has_local_layer',
                                                                           value_range=(True,),
                                                                           default_value=True)
                config_space = available_decoders[decoder_name].get_hyperparameter_search_space(dataset_properties,
                                                                                                # type: ignore
                                                                                                **updates)
                compatible_encoders = decoder2encoder[decoder_name]
                encoders_with_multi_decoder = []
                encoder_with_single_decoder = []
                for encoder in compatible_encoders:
                    if len(encoder2decoder[encoder]) > 1:
                        encoders_with_multi_decoder.append(encoder)
                    else:
                        encoder_with_single_decoder.append(encoder)

                cs.add_configuration_space(
                    block_prefix + decoder_name,
                    config_space,
                    # parent_hyperparameter=parent_hyperparameter
                )

                hps = cs.get_hyperparameters()  # type: List[CSH.Hyperparameter]
                conditions_to_add = []
                for hp in hps:
                    # TODO consider if this will raise any unexpected behavior
                    if hp.name.startswith(block_prefix + decoder_name):
                        # From the implementation of ConfigSpace
                        # Only add a condition if the parameter is a top-level
                        # parameter of the new configuration space (this will be some
                        #  kind of tree structure).
                        if cs.get_parents_of(hp):
                            continue
                        or_cond = []
                        for encoder_single in encoder_with_single_decoder:
                            or_cond.append(EqualsCondition(hp,
                                                           hp_encoder,
                                                           encoder_single))
                        for encode_multi in encoders_with_multi_decoder:
                            hp_decoder_type = cs.get_hyperparameter(f'{block_prefix + encode_multi}:decoder_type')
                            or_cond.append(EqualsCondition(hp, hp_decoder_type, decoder_name))
                        if len(or_cond) == 0:
                            continue
                        elif len(or_cond) > 1:
                            conditions_to_add.append(OrConjunction(*or_cond))
                        else:
                            conditions_to_add.append(or_cond[0])

                cs.add_conditions(conditions_to_add)

        use_temporal_fusion = get_hyperparameter(use_temporal_fusion, CategoricalHyperparameter)
        cs.add_hyperparameter(use_temporal_fusion)
        if True in use_temporal_fusion.choices:
            update = self._get_search_space_updates(prefix=self.tf_prefix)
            cs_tf = TemporalFusion.get_hyperparameter_search_space(dataset_properties,
                                                                   **update)
            parent_hyperparameter = {'parent': use_temporal_fusion, 'value': True}
            cs.add_configuration_space(
                self.tf_prefix,
                cs_tf,
                parent_hyperparameter=parent_hyperparameter
            )

        for encoder_name, encoder in available_encoders.items():
            encoder_is_casual = encoder.encoder_properties()
            if not encoder_is_casual:
                # we do not allow non-casual encoder to appear in the lower layer of the network. e.g, if we have an
                # encoder with 3 blocks, then non_casual encoder is only allowed to appear in the third layer
                for i in range(max(min_num_blocks, 2), max_num_blocks + 1):
                    for j in range(1, i):
                        choice_hp = cs.get_hyperparameter(f"block_{j}:__choice__")
                        if encoder_name in choice_hp.choices:
                            forbidden_encoder_uncasual = [ForbiddenEqualsClause(num_blocks, i),
                                                          ForbiddenEqualsClause(choice_hp, encoder_name)]
                            if forbidden_decoder_ar is not None:
                                forbidden_encoder_uncasual.append(forbidden_decoder_ar)
                            forbiddens_decoder_auto_regressive.append(
                                ForbiddenAndConjunction(*forbidden_encoder_uncasual)
                            )

        cs.add_forbidden_clauses(forbiddens_decoder_auto_regressive)

        if self.deepAR_decoder_name in available_decoders:
            deep_ar_hp = ':'.join([self.deepAR_decoder_prefix, self.deepAR_decoder_name, 'auto_regressive'])
            if deep_ar_hp in cs:
                deep_ar_hp = cs.get_hyperparameter(deep_ar_hp)
                forbidden_deep_ar = ForbiddenEqualsClause(deep_ar_hp, True)
                if min_num_blocks == 1:
                    if max_num_blocks > 1:
                        if max_num_blocks - min_num_blocks > 1:
                            forbidden = ForbiddenAndConjunction(
                                ForbiddenInClause(num_blocks, list(range(1, max_num_blocks))),
                                forbidden_deep_ar
                            )
                        else:
                            forbidden = ForbiddenAndConjunction(ForbiddenEqualsClause(num_blocks, 2), forbidden_deep_ar)
                        cs.add_forbidden_clause(forbidden)

                forbidden_deep_ars = []

                hps_forbidden_deep_ar = [variable_selection, use_temporal_fusion]
                for hp_forbidden_deep_ar in hps_forbidden_deep_ar:
                    if True in hp_forbidden_deep_ar.choices:
                        forbidden_deep_ars.append(ForbiddenAndConjunction(
                            ForbiddenEqualsClause(hp_forbidden_deep_ar, True),
                            forbidden_deep_ar
                        ))
                if forbidden_deep_ars:
                    cs.add_forbidden_clauses(forbidden_deep_ars)

        return cs

    def set_hyperparameters(self,
                            configuration: Configuration,
                            init_params: Optional[Dict[str, Any]] = None
                            ) -> 'autoPyTorchChoice':
        """
        Applies a configuration to the given component.
        This method translate a hierarchical configuration key,
        to an actual parameter of the autoPyTorch component.

        Args:
            configuration (Configuration):
                Which configuration to apply to the chosen component
            init_params (Optional[Dict[str, any]]):
                Optional arguments to initialize the chosen component

        Returns:
            self: returns an instance of self
        """

        params = configuration.get_dictionary()
        num_blocks = params['num_blocks']
        decoder_auto_regressive = params['decoder_auto_regressive']
        use_temporal_fusion = params['use_temporal_fusion']
        forecasting_structure_kwargs = dict(num_blocks=num_blocks,
                                            use_temporal_fusion=use_temporal_fusion,
                                            variable_selection=params['variable_selection'],
                                            skip_connection=params['skip_connection'])
        if 'share_single_variable_networks' in params:
            forecasting_structure_kwargs['share_single_variable_networks'] = params['share_single_variable_networks']
            del params['share_single_variable_networks']

        del params['num_blocks']
        del params['use_temporal_fusion']
        del params['variable_selection']
        del params['skip_connection']
        del params['decoder_auto_regressive']

        if 'skip_connection_type' in params:
            forecasting_structure_kwargs['skip_connection_type'] = params['skip_connection_type']
            del params['skip_connection_type']
            if 'grn_use_dropout' in params:
                del params['grn_use_dropout']
                if 'grn_dropout_rate' in params:
                    forecasting_structure_kwargs['grn_dropout_rate'] = params['grn_dropout_rate']
                    del params['grn_dropout_rate']
                else:
                    forecasting_structure_kwargs['grn_dropout_rate'] = 0.0

        pipeline_steps = [('net_structure', ForecastingNetworkStructure(**forecasting_structure_kwargs))]
        self.encoder_choice = []
        self.decoder_choice = []

        decoder_components = self.get_decoder_components()

        for i in range(1, num_blocks + 1):
            new_params = {}

            block_prefix = f'block_{i}:'
            choice = params[block_prefix + '__choice__']
            del params[block_prefix + '__choice__']

            for param, value in params.items():
                if param.startswith(block_prefix):
                    param = param.replace(block_prefix + choice + ':', '')
                    new_params[param] = value

            if init_params is not None:
                for param, value in init_params.items():
                    if param.startswith(block_prefix):
                        param = param.replace(block_prefix + choice + ':', '')
                        new_params[param] = value

            decoder_type = None

            decoder_params = {}
            decoder_params_names = []
            for param, value in new_params.items():
                if decoder_type is None:
                    for decoder_component in decoder_components.keys():
                        if param.startswith(block_prefix + decoder_component):
                            decoder_type = decoder_component
                            decoder_params_names.append(param)
                            param = param.replace(block_prefix + decoder_type + ':', '')
                            decoder_params[param] = value
                else:
                    if param.startswith(block_prefix + decoder_type):
                        decoder_params_names.append(param)
                        param = param.replace(block_prefix + decoder_type + ':', '')
                        decoder_params[param] = value

            for param_name in decoder_params_names:
                del new_params[param_name]
            new_params['random_state'] = self.random_state
            new_params['block_number'] = i
            decoder_params['random_state'] = self.random_state
            decoder_params['block_number'] = i
            # for mlp decoder, to avoid decoder's auto_regressive being overwritten by decoder_auto_regressive
            if 'auto_regressive' not in decoder_params:
                decoder_params['auto_regressive'] = decoder_auto_regressive
            encoder = self.get_components()[choice](**new_params)
            decoder = decoder_components[decoder_type](**decoder_params)
            pipeline_steps.extend([(f'encoder_{i}', encoder), (f'decoder_{i}', decoder)])
            self.encoder_choice.append(encoder)
            self.decoder_choice.append(decoder)

        new_params = {}
        if use_temporal_fusion:
            for param, value in params.items():
                if param.startswith(self.tf_prefix):
                    param = param.replace(self.tf_prefix + ':', '')
                    new_params[param] = value
            temporal_fusion = TemporalFusion(self.random_state,
                                             **new_params)
            pipeline_steps.extend([(f'temporal_fusion', temporal_fusion)])

        self.pipeline = Pipeline(pipeline_steps)
        self.choice = self.encoder_choice[0]
        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'SeqEncoder',
            'name': 'SeqEncoder',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }
