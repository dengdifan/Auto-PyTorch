from typing import List

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, \
    UniformFloatHyperparameter
from ConfigSpace.conditions import GreaterThanCondition, EqualsCondition, AndConjunction

from typing import Dict, Optional, Tuple, Union, Any

from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_head.utils import _activations
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, get_hyperparameter

from autoPyTorch.pipeline.components.setup.network_backbone.\
    forecasting_backbone.forecasting_decoder.base_forecasting_decoder import BaseForecastingDecoder, DecoderProperties


class NBEATSBLock(nn.Module):
    def __init__(self,
                 n_in_features: int,
                 stack_idx: int,
                 stack_type: str,
                 num_blocks: int,
                 num_layers: int,
                 width: int,
                 normalization: str,
                 activation: str,
                 weight_sharing: bool,
                 expansion_coefficient_length: int,
                 use_dropout: bool,
                 dropout_rate: Optional[float] = None,
                 ):
        super().__init__()
        self.n_in_features = n_in_features
        self.stack_idx = stack_idx
        self.stack_type = stack_type

        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.width = width
        self.normalization = normalization
        self.activation = activation
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self.expansion_coefficient_length = expansion_coefficient_length

        self.weight_sharing = weight_sharing

        self.backbone = nn.Sequential(*self.build_backbone())

        self.backcast_head = None
        self.forecast_head = None

    def build_backbone(self):
        layers: List[nn.Module] = list()
        n_in_features = self.n_in_features
        for _ in range(self.num_layers):
            self._add_layer(layers, n_in_features)
            n_in_features = self.width
        return layers

    def _add_layer(self, layers: List[nn.Module], in_features: int) -> None:
        layers.append(nn.Linear(in_features, self.width))
        if self.normalization == 'BN':
            layers.append(nn.BatchNorm1d(self.width))
        elif self.normalization == 'LN':
            layers.append(nn.LayerNorm(self.width))
        layers.append(_activations[self.activation]())
        if self.use_dropout:
            layers.append(nn.Dropout(self.dropout_rate))

    def forward(self, x):
        if self.backcast_head is None and self.forecast_head is None:
            # used to compute head dimensions
            return self.backbone(x)
        else:
            x = self.backbone(x)
            forecast = self.forecast_head(x)
            backcast = self.backcast_head(x)
            return backcast, forecast


class NBEATSDecoder(BaseForecastingDecoder):
    _fixed_seq_length = True
    window_size = 1
    fill_lower_resolution_seq = False
    fill_kwargs = {}

    @staticmethod
    def decoder_properties() -> DecoderProperties:
        return DecoderProperties(multi_blocks=True)

    def _build_decoder(self,
                       encoder_output_shape: Tuple[int, ...],
                       future_variable_input: Tuple[int, ...],
                       n_prediction_heads: int,
                       dataset_properties: Dict) -> Tuple[nn.Module, int]:
        in_features = encoder_output_shape[-1]
        n_beats_type = self.config['n_beats_type']
        if n_beats_type == 'G':
            stacks = [[] for _ in range(self.config['num_stacks_g'])]
            for stack_idx in range(1, self.config['num_stacks_g'] + 1):
                for block_idx in range(self.config['num_blocks_g']):
                    if self.config['weight_sharing_g'] and block_idx > 0:
                        # for weight sharing, we only create one instance
                        break
                    ecl = self.config['expansion_coefficient_length_g']
                    stacks[stack_idx - 1].append(NBEATSBLock(in_features,
                                                             stack_idx=stack_idx,
                                                             stack_type='generic',
                                                             num_blocks=self.config['num_blocks_g'],
                                                             num_layers=self.config['num_layers_g'],
                                                             width=self.config['width_g'],
                                                             normalization=self.config['normalization'],
                                                             activation=self.config['activation'],
                                                             weight_sharing=self.config['weight_sharing_g'],
                                                             expansion_coefficient_length=ecl,
                                                             use_dropout=self.config['use_dropout_g'],
                                                             dropout_rate=self.config.get('dropout_g', None),
                                                             ))

        elif n_beats_type == 'I':
            stacks = [[] for _ in range(self.config['num_stacks_i'])]
            for stack_idx in range(1, self.config['num_stacks_i'] + 1):
                for block_idx in range(self.config['num_blocks_i_%d' % stack_idx]):
                    if self.config['weight_sharing_i_%d' % stack_idx] and block_idx > 0:
                        # for weight sharing, we only create one instance
                        break
                    stack_type = self.config['stack_type_i_%d' % stack_idx]
                    if stack_type == 'generic':
                        ecl = self.config['expansion_coefficient_length_i_generic_%d' % stack_idx]
                    elif stack_type == 'trend':
                        ecl = self.config['expansion_coefficient_length_i_trend_%d' % stack_idx]
                    elif stack_type == 'seasonality':
                        ecl = self.config['expansion_coefficient_length_i_seasonality_%d' % stack_idx]
                    else:
                        raise ValueError(f"Unsupported stack_type {stack_type}")

                    stacks[stack_idx - 1].append(NBEATSBLock(in_features,
                                                             stack_idx=stack_idx,
                                                             stack_type=stack_type,
                                                             num_blocks=self.config['num_blocks_i_%d' % stack_idx],
                                                             num_layers=self.config['num_layers_i_%d' % stack_idx],
                                                             width=self.config['width_i_%d' % stack_idx],
                                                             normalization=self.config['normalization'],
                                                             activation=self.config['activation'],
                                                             weight_sharing=self.config[f'weight_sharing_i_%d' %
                                                                                        stack_idx],
                                                             expansion_coefficient_length=ecl,
                                                             use_dropout=self.config['use_dropout_i'],
                                                             dropout_rate=self.config.get('dropout_i_%d' %
                                                                                          stack_idx, None),
                                                             ))
        else:
            raise ValueError(f"Unsupported n_beats_type: {n_beats_type}")
        return stacks, stacks[-1][-1].width

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'NBEATSDecoder',
            'name': 'NBEATSDecoder',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    @property
    def fitted_encoder(self):
        return ['NBEATSEncoder']

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        X.update({'backcast_loss_ratio': self.config['backcast_loss_ratio']})
        return super().transform(X)

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
            n_beats_type: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="n_beats_type",
                value_range=('I', 'G'),
                default_value='I'
            ),
            num_stacks_g: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="num_stacks_g",
                value_range=(2, 32),
                default_value=30,
                log=True,
            ),
            num_blocks_g: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'num_blocks_g',
                value_range=(1, 2),
                default_value=1
            ),
            num_layers_g: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'num_layers_g',
                value_range=(1, 4),
                default_value=4
            ),
            width_g: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'width_g',
                value_range=(16, 512),
                default_value=256,
                log=True
            ),
            num_stacks_i: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="num_stacks_i",
                value_range=(1, 4),
                default_value=2
            ),
            num_blocks_i: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'num_blocks_i',
                value_range=(1, 5),
                default_value=3
            ),
            num_layers_i: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'num_layers_i',
                value_range=(1, 5),
                default_value=3
            ),
            width_i: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'width_i',
                value_range=(16, 2048),
                default_value=512,
                log=True
            ),
            weight_sharing: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'weight_sharing',
                value_range=(True, False),
                default_value=False,
            ),
            stack_type: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'stack_type',
                value_range=('generic', 'seasonality', 'trend'),
                default_value='generic'),
            expansion_coefficient_length_generic: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'expansion_coefficient_length_generic',
                value_range=(16, 64),
                default_value=32,
                log=True
            ),
            expansion_coefficient_length_seasonality: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'expansion_coefficient_length_seasonality',
                value_range=(1, 8),
                default_value=3,
            ),
            expansion_coefficient_length_trend: HyperparameterSearchSpace = HyperparameterSearchSpace(
                'expansion_coefficient_length_trend',
                value_range=(1, 4),
                default_value=3,
            ),
            activation: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="activation",
                value_range=tuple(_activations.keys()),
                default_value=list(_activations.keys())[0],
            ),
            use_dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="use_dropout",
                value_range=(True, False),
                default_value=False,
            ),
            normalization: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="normalization",
                value_range=('BN', 'LN', 'NoNorm'),
                default_value='BN'
            ),
            dropout: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="dropout",
                value_range=(0, 0.8),
                default_value=0.1,
            ),
            backcast_loss_ratio: HyperparameterSearchSpace = HyperparameterSearchSpace(
                hyperparameter="backcast_loss_ratio",
                value_range=(0., 1.),
                default_value=1.,
            )
    ) -> ConfigurationSpace:
        """
        Configuration for N-BEATS. The network is composed of several stacks, each stack is composed of several block,
        we follow the implementation from N-BEATS: blocks are only composed of fully-connected layers with the same
        width
        The design of the configuration space follows pytorch-forecasting:
        https://github.com/jdb78/pytorch-forecasting/tree/master/pytorch_forecasting/models/nbeats
        Give that N-BEATS-I and N-BEATS-G's default hyperparameter configuration that totally different, we consider
        them as two seperate configuration space: N-BEATS-G that only contains generic blocks and thus could be scaled
        up to 32 stacks, while each stacks share the same number of blocks/ width/ dropout rate. While N-BEATS-I is
        is restricted to be a network with a much smaller number of stacks. However, the block type of N-BEATS-G at each
        stack can be freely selected
        freely selected
        Args:
            dataset_properties:
            n_beats_type: type of nbeats network, could be I (N-BEATS-I) or G (N-BEATS-G)
            num_stacks_g: number of stacks
            num_blocks_g: number of blocks per stack
            num_layers_g: number of fc layers per block, this value is the same across all the blocks within one stack
            width_g: fc layer width, this value is the same across all the blocks within one stack
            num_stacks_i: number of stacks
            num_blocks_i: number of blocks per stack
            num_layers_i: number of fc layers per block, this value is the same across all the blocks within one stack
            width_i: fc layer width, this value is the same across all the blocks within one stack
            weight_sharing: if weights are shared inside one block
            stack_type: stack type, used to define the final output
            expansion_coefficient_length_generic: expansion_coefficient_length, activate if stack_type is 'generic'
            expansion_coefficient_length_seasonality: expansion_coefficient_length, activate if stack_type is
                'seasonality' (n_dim = expansion_coefficient_length_interpretable * n_prediciton_steps)
            expansion_coefficient_length_trend: expansion_coefficient_length, activate if stack_type is 'trend' (it
                corresponds to the degree of the polynomial)
            activation: activation function across fc layers
            use_dropout: if dropout is applied
            normalization: if normalization is applied
            dropout: dropout value, if use_dropout is set as True
            backcast_loss_ratio: weight of backcast in comparison to forecast when calculating the loss.
                A weight of 1.0 means that forecast and backcast loss is weighted the same (regardless of backcast and
                forecast lengths). Defaults to 0.0, i.e. no weight.
        Returns:
            Configuration Space
        """

        cs = ConfigurationSpace()

        n_beats_type = get_hyperparameter(n_beats_type, CategoricalHyperparameter)

        # General Hyperparameters
        add_hyperparameter(cs, activation, CategoricalHyperparameter)
        add_hyperparameter(cs, normalization, CategoricalHyperparameter)
        add_hyperparameter(cs, backcast_loss_ratio, UniformFloatHyperparameter)

        cs.add_hyperparameter(n_beats_type)
        # N-BEATS-G

        weight_sharing_g = HyperparameterSearchSpace(hyperparameter='weight_sharing_g',
                                                     value_range=weight_sharing.value_range,
                                                     default_value=weight_sharing.default_value,
                                                     log=weight_sharing.log)
        use_dropout_g = HyperparameterSearchSpace(hyperparameter='use_dropout_g',
                                                  value_range=use_dropout.value_range,
                                                  default_value=use_dropout.default_value,
                                                  log=use_dropout.log)
        dropout_g = HyperparameterSearchSpace(hyperparameter='dropout_g',
                                              value_range=dropout.value_range,
                                              default_value=dropout.default_value,
                                              log=dropout.log)
        ecl_g_search_space = HyperparameterSearchSpace(
            hyperparameter='expansion_coefficient_length_g',
            value_range=expansion_coefficient_length_generic.value_range,
            default_value=expansion_coefficient_length_generic.default_value,
            log=expansion_coefficient_length_generic.log
        )

        num_stacks_g = get_hyperparameter(num_stacks_g, UniformIntegerHyperparameter)
        num_blocks_g = get_hyperparameter(num_blocks_g, UniformIntegerHyperparameter)
        num_layers_g = get_hyperparameter(num_layers_g, UniformIntegerHyperparameter)
        width_g = get_hyperparameter(width_g, UniformIntegerHyperparameter)
        weight_sharing_g = get_hyperparameter(weight_sharing_g, CategoricalHyperparameter)
        ecl_g = get_hyperparameter(ecl_g_search_space, UniformIntegerHyperparameter)
        use_dropout_g = get_hyperparameter(use_dropout_g, CategoricalHyperparameter)

        dropout_g = get_hyperparameter(dropout_g, UniformFloatHyperparameter)

        n_beats_g_hps = [num_stacks_g, num_blocks_g, num_layers_g, width_g, weight_sharing_g, ecl_g, use_dropout_g]
        n_beats_g_conds = [EqualsCondition(hp_nbeats_g, n_beats_type, 'G') for hp_nbeats_g in n_beats_g_hps]
        cs.add_hyperparameters(n_beats_g_hps)
        cs.add_hyperparameter(dropout_g)
        cs.add_conditions(n_beats_g_conds)
        cs.add_condition(AndConjunction(EqualsCondition(dropout_g, n_beats_type, 'G'),
                                        EqualsCondition(dropout_g, use_dropout_g, True)))

        min_num_stacks_i, max_num_stacks_i = num_stacks_i.value_range

        use_dropout_i = HyperparameterSearchSpace(hyperparameter='use_dropout_i',
                                                  value_range=use_dropout.value_range,
                                                  default_value=use_dropout.default_value,
                                                  log=use_dropout.log)

        num_stacks_i = get_hyperparameter(num_stacks_i, UniformIntegerHyperparameter)
        use_dropout_i = get_hyperparameter(use_dropout_i, CategoricalHyperparameter)

        cs.add_hyperparameters([num_stacks_i, use_dropout_i])
        cs.add_conditions([EqualsCondition(num_stacks_i, n_beats_type, 'I'),
                           EqualsCondition(use_dropout_i, n_beats_type, 'I')
                           ])

        for stack_idx in range(1, int(max_num_stacks_i) + 1):
            num_blocks_i_search_space = HyperparameterSearchSpace(hyperparameter='num_blocks_i_%d' % stack_idx,
                                                                  value_range=num_blocks_i.value_range,
                                                                  default_value=num_blocks_i.default_value,
                                                                  log=num_blocks_i.log)
            num_layers_i_search_space = HyperparameterSearchSpace(hyperparameter='num_layers_i_%d' % stack_idx,
                                                                  value_range=num_layers_i.value_range,
                                                                  default_value=num_layers_i.default_value,
                                                                  log=num_layers_i.log)
            width_i_search_space = HyperparameterSearchSpace(hyperparameter='width_i_%d' % stack_idx,
                                                             value_range=width_i.value_range,
                                                             default_value=width_i.default_value,
                                                             log=width_i.log)
            weight_sharing_i_search_space = HyperparameterSearchSpace(hyperparameter='weight_sharing_i_%d' % stack_idx,
                                                                      value_range=weight_sharing.value_range,
                                                                      default_value=weight_sharing.default_value,
                                                                      log=weight_sharing.log)
            stack_type_i_search_space = HyperparameterSearchSpace(hyperparameter='stack_type_i_%d' % stack_idx,
                                                                  value_range=stack_type.value_range,
                                                                  default_value=stack_type.default_value,
                                                                  log=stack_type.log)
            expansion_coefficient_length_generic_search_space = HyperparameterSearchSpace(
                hyperparameter='expansion_coefficient_length_i_generic_%d' % stack_idx,
                value_range=expansion_coefficient_length_generic.value_range,
                default_value=expansion_coefficient_length_generic.default_value,
                log=expansion_coefficient_length_generic.log
            )
            expansion_coefficient_length_seasonality_search_space = HyperparameterSearchSpace(
                hyperparameter='expansion_coefficient_length_i_seasonality_%d' % stack_idx,
                value_range=expansion_coefficient_length_seasonality.value_range,
                default_value=expansion_coefficient_length_seasonality.default_value,
                log=expansion_coefficient_length_seasonality.log
            )
            expansion_coefficient_length_trend_search_space = HyperparameterSearchSpace(
                hyperparameter='expansion_coefficient_length_i_trend_%d' % stack_idx,
                value_range=expansion_coefficient_length_trend.value_range,
                default_value=expansion_coefficient_length_trend.default_value,
                log=expansion_coefficient_length_trend.log
            )

            num_blocks_i_hp = get_hyperparameter(num_blocks_i_search_space, UniformIntegerHyperparameter)
            num_layers_i_hp = get_hyperparameter(num_layers_i_search_space, UniformIntegerHyperparameter)
            width_i_hp = get_hyperparameter(width_i_search_space, UniformIntegerHyperparameter)
            weight_sharing_i_hp = get_hyperparameter(weight_sharing_i_search_space, CategoricalHyperparameter)
            stack_type_i_hp = get_hyperparameter(stack_type_i_search_space, CategoricalHyperparameter)

            expansion_coefficient_length_generic_hp = get_hyperparameter(
                expansion_coefficient_length_generic_search_space,
                UniformIntegerHyperparameter
            )
            expansion_coefficient_length_seasonality_hp = get_hyperparameter(
                expansion_coefficient_length_seasonality_search_space,
                UniformIntegerHyperparameter
            )
            expansion_coefficient_length_trend_hp = get_hyperparameter(
                expansion_coefficient_length_trend_search_space,
                UniformIntegerHyperparameter
            )

            hps = [num_blocks_i_hp, num_layers_i_hp, width_i_hp, stack_type_i_hp, weight_sharing_i_hp]
            cs.add_hyperparameters([*hps,
                                    expansion_coefficient_length_generic_hp,
                                    expansion_coefficient_length_seasonality_hp,
                                    expansion_coefficient_length_trend_hp])

            cond_ecls = [
                EqualsCondition(expansion_coefficient_length_generic_hp, stack_type_i_hp, 'generic'),
                EqualsCondition(expansion_coefficient_length_seasonality_hp, stack_type_i_hp, 'seasonality'),
                EqualsCondition(expansion_coefficient_length_trend_hp, stack_type_i_hp, 'trend'),
            ]

            if stack_idx > int(min_num_stacks_i):
                # The units of layer i should only exist
                # if there are at least i layers
                for hp in hps:
                    cs.add_condition(
                        AndConjunction(GreaterThanCondition(hp, num_stacks_i, stack_idx - 1),
                                       EqualsCondition(hp, n_beats_type, 'I'))
                    )
                for cond_ecl in cond_ecls:
                    cs.add_condition(
                        AndConjunction(cond_ecl,
                                       GreaterThanCondition(cond_ecl.child, num_stacks_i, stack_idx - 1),
                                       EqualsCondition(cond_ecl.child, n_beats_type, 'I'))
                    )
            else:
                cs.add_conditions([EqualsCondition(hp, n_beats_type, 'I') for hp in hps])
                cs.add_conditions([
                    AndConjunction(cond_ecl,
                                   EqualsCondition(cond_ecl.child, n_beats_type, 'I')) for cond_ecl in cond_ecls
                ]
                )

            dropout_search_space = HyperparameterSearchSpace(hyperparameter='dropout_i_%d' % stack_idx,
                                                             value_range=dropout.value_range,
                                                             default_value=dropout.default_value,
                                                             log=dropout.log)

            dropout_hp = get_hyperparameter(dropout_search_space, UniformFloatHyperparameter)
            cs.add_hyperparameter(dropout_hp)

            dropout_condition_1 = EqualsCondition(dropout_hp, use_dropout_i, True)
            dropout_condition_2 = EqualsCondition(dropout_hp, n_beats_type, 'I')

            if stack_idx > int(min_num_stacks_i):
                dropout_condition_3 = GreaterThanCondition(dropout_hp, num_stacks_i, stack_idx - 1)
                cs.add_condition(AndConjunction(dropout_condition_1, dropout_condition_2, dropout_condition_3))
            else:
                cs.add_condition(AndConjunction(dropout_condition_1, dropout_condition_2))

        return cs