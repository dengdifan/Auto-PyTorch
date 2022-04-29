from typing import Any, Dict, Optional, List, Tuple, Set, Union

import torch
from torch import nn

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.components import (
    EncoderBlockInfo, EncoderOutputForm
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.components import (
    DecoderBlockInfo
)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import (
    NetworkStructure,
    AddLayer
)

from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    GateAddNorm, GatedResidualNetwork, VariableSelectionNetwork, InterpretableMultiHeadAttention
)


class TemporalFusionLayer(nn.Module):
    """
    (Lim et al.
    Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting,
    https://arxiv.org/abs/1912.09363)
    we follow the implementation from pytorch forecasting:
    https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/models/temporal_fusion_transformer/__init__.py
    """

    def __init__(self,
                 window_size: int,
                 network_structure: NetworkStructure,
                 network_encoder: Dict[str, EncoderBlockInfo],
                 n_decoder_output_features: int,
                 d_model: int,
                 n_head: int,
                 dropout: Optional[float] = None):
        super().__init__()
        num_blocks = network_structure.num_blocks
        last_block = f'block_{num_blocks}'
        n_encoder_output = network_encoder[last_block].encoder_output_shape[-1]
        self.window_size = window_size

        if n_decoder_output_features != n_encoder_output:
            self.decoder_proj_layer = nn.Linear(n_decoder_output_features, n_encoder_output, bias=False)
        else:
            self.decoder_proj_layer = None
        if network_structure.variable_selection:
            if network_structure.skip_connection:
                # static feature selector needs to generate the same number of features as the output of the encoder
                n_encoder_output_first = network_encoder['block_1'].encoder_output_shape[-1]
                self.static_context_enrichment = GatedResidualNetwork(
                    n_encoder_output_first, n_encoder_output_first, n_encoder_output_first, dropout
                )
                self.enrichment = GatedResidualNetwork(
                    input_size=n_encoder_output,
                    hidden_size=n_encoder_output,
                    output_size=d_model,
                    dropout=dropout,
                    context_size=n_encoder_output_first,
                    residual=False,
                )
                self.enrich_with_static = True
        if not hasattr(self, 'enrichment'):
            self.enrichment = GatedResidualNetwork(
                input_size=n_encoder_output,
                hidden_size=n_encoder_output,
                output_size=d_model,
                dropout=dropout,
                residual=False,
            )
            self.enrich_with_static = False

        self.attention_fusion = InterpretableMultiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            dropout=dropout or 0.0
        )
        self.post_attn_gate_norm = GateAddNorm(d_model, dropout=dropout, trainable_add=False)
        self.pos_wise_ff = GatedResidualNetwork(input_size=d_model, hidden_size=d_model,
                                                output_size=d_model, dropout=dropout)

        self.network_structure = network_structure
        if network_structure.skip_connection:
            if network_structure.skip_connection_type == 'add':
                self.residual_connection = AddLayer(d_model, n_encoder_output)
            elif network_structure.skip_connection_type == 'gate_add_norm':
                self.residual_connection = GateAddNorm(d_model, skip_size=n_encoder_output,
                                                       dropout=None, trainable_add=False)

    def forward(self,
                encoder_output: torch.Tensor,
                decoder_output: torch.Tensor,
                past_observed_values: torch.BoolTensor,
                decoder_length: int,
                static_embedding: Optional[torch.Tensor] = None):
        """
        Args:
            encoder_output: the output of the last layer of encoder network
            decoder_output: the output of the last layer of decoder network
            past_observed_values: observed values in the past
            decoder_length: length of decoder network
            static_embedding: output of static variable selection network (if applible)
        """
        if self.decoder_proj_layer is not None:
            decoder_output = self.decoder_proj_layer(decoder_output)

        network_output = torch.cat([encoder_output, decoder_output], dim=1)

        if self.enrich_with_static:
            static_context_enrichment = self.static_context_enrichment(static_embedding)
            attn_input = self.enrichment(
                network_output, static_context_enrichment[:, None].expand(-1, network_output.shape[1], -1)
            )
        else:
            attn_input = self.enrichment(network_output)

        # Attention
        encoder_out_length = encoder_output.shape[1]
        past_observed_values = past_observed_values[:, -encoder_out_length:]
        past_observed_values = past_observed_values.to(self.device)

        mask = self.get_attention_mask(past_observed_values=past_observed_values, decoder_length=decoder_length)
        if mask.shape[-1] < attn_input.shape[1]:
            # in case that none of the samples has length greater than window_size
            mask = torch.cat([
                mask.new_full((*mask.shape[:-1], attn_input.shape[1] - mask.shape[-1]), True),
                mask
            ], dim=-1)

        attn_output, attn_output_weights = self.attention_fusion(
            q=attn_input[:, -decoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=mask)

        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, -decoder_length:])
        output = self.pos_wise_ff(attn_output)

        if self.network_structure.skip_connection:
            return self.residual_connection(output, decoder_output)
        else:
            return output

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self.to(device)
        self._device = device

    def get_attention_mask(self, past_observed_values: torch.BoolTensor, decoder_length: int):
        """
        https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/models/
        temporal_fusion_transformer/__init__.py
        """
        # indices to which is attended
        attend_step = torch.arange(decoder_length, device=self.device)
        # indices for which is predicted
        predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
        # do not attend to steps to self or after prediction
        # todo: there is potential value in attending to future forecasts if they are made with knowledge currently
        #   available
        #   one possibility is here to use a second attention layer for future attention (assuming different effects
        #   matter in the future than the past)
        #   or alternatively using the same layer but allowing forward attention - i.e. only masking out non-available
        #   data and self
        decoder_mask = attend_step >= predict_step
        # do not attend to steps where data is padded
        # this is the result of our padding strategy: we pad values at the start of the tensors
        encoder_mask = ~past_observed_values.squeeze(-1)

        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(encoder_mask.size(0), -1, -1),
            ),
            dim=2,
        )
        return mask


class VariableSelector(nn.Module):
    def __init__(self,
                 network_structure: NetworkStructure,
                 dataset_properties: Dict[str, Any],
                 network_encoder: Dict[str, EncoderBlockInfo],
                 auto_regressive: bool = False,
                 feature_names: Tuple[str] = (),
                 known_future_features: Tuple[str] = tuple(),
                 feature_shapes: Dict[str, int] = {},
                 static_features: Tuple[Union[str, int]] = (),
                 time_feature_names: Tuple[str] = (),
                 ):
        """
        Variable Selector. This models follows the implementation from
        pytorch_forecasting.models.temporal_fusion_transformer.sub_modules.VariableSelectionNetwork
        However, we adjust the structure to fit the data extracted from our dataloader: we record the feature index from
        each feature names and break the input features on the fly.

        The order of the input variables is as follows:
        [features (from the dataset), time_features (from time feature transformers), targets]
        Args:
            network_structure (NetworkStructure): contains the information of the overall architecture information
            dataset_properties (Dict): dataset properties
            network_encoder(Dict[str, EncoderBlockInfo]): Network encoders
            auto_regressive bool: if it belongs to an auto-regressive model
            feature_names Tuple[str]: feature names, used to construct the selection network
            known_future_features Tuple[str]: known future features
            feature_shapes Dict[str, int]: shapes of each features
            time_feature_names Tuple[str]: time feature names, used to complement feature_shapes
        """
        super().__init__()
        first_encoder_output_shape = network_encoder['block_1'].encoder_output_shape[-1]
        self.hidden_size = first_encoder_output_shape

        assert set(feature_names) == set(feature_shapes.keys()), f"feature_names and feature_shapes must have " \
                                                                 f"the same variable names but they are different" \
                                                                 f"at {set(feature_names) ^ set(feature_shapes.keys())}"
        pre_scalar = {'past_targets': nn.Linear(dataset_properties['output_shape'][-1], self.hidden_size)}
        encoder_input_sizes = {'past_targets': self.hidden_size}
        decoder_input_sizes = {}
        future_feature_name2tensor_idx = {}
        feature_names2tensor_idx = {}
        idx_tracker = 0
        idx_tracker_future = 0

        static_features = set(static_features)
        static_features_input_size = {}

        # static_features should always be known beforehand
        known_future_features = tuple(known_future_features)

        if feature_names:
            for name in feature_names:
                feature_shape = feature_shapes[name]
                feature_names2tensor_idx[name] = [idx_tracker, idx_tracker + feature_shape]
                idx_tracker += feature_shape
                pre_scalar[name] = nn.Linear(feature_shape, self.hidden_size)
                if name in static_features:
                    static_features_input_size[name] = self.hidden_size
                else:
                    encoder_input_sizes[name] = self.hidden_size
                    if name in known_future_features:
                        decoder_input_sizes[name] = self.hidden_size

        for future_name in known_future_features:
            feature_shape = feature_shapes[future_name]
            future_feature_name2tensor_idx[future_name] = [idx_tracker_future, idx_tracker_future + feature_shape]
            idx_tracker_future += feature_shape


        if time_feature_names:
            for name in time_feature_names:
                feature_names2tensor_idx[name] = [idx_tracker, idx_tracker+1]
                future_feature_name2tensor_idx[name] = [idx_tracker_future, idx_tracker_future + 1]
                idx_tracker += 1
                idx_tracker_future += 1
                pre_scalar[name] = nn.Linear(1, self.hidden_size)
                encoder_input_sizes[name] = self.hidden_size
                decoder_input_sizes[name] = self.hidden_size

        if not feature_names or not known_future_features:
            # Ensure that at least one feature is applied
            placeholder_features = 'placeholder_features'
            i = 0

            self.placeholder_features = []
            while placeholder_features in feature_names or placeholder_features in self.placeholder_features:
                i += 1
                placeholder_features = f'placeholder_features_{i}'
                if i == 5000:
                    raise RuntimeError(
                        "Cannot assign name to placeholder features, please considering rename your features")

            name = placeholder_features
            pre_scalar[name] = nn.Linear(1, self.hidden_size)
            encoder_input_sizes[name] = self.hidden_size
            decoder_input_sizes[name] = self.hidden_size
            self.placeholder_features.append(placeholder_features)

        feature_names = time_feature_names + feature_names
        known_future_features = time_feature_names + known_future_features

        self.feature_names = feature_names
        self.feature_names2tensor_idx = feature_names2tensor_idx
        self.future_feature_name2tensor_idx = future_feature_name2tensor_idx
        self.known_future_features = known_future_features

        if auto_regressive:
            pre_scalar.update({'future_prediction': nn.Linear(dataset_properties['output_shape'][-1],
                                                              self.hidden_size)})
            decoder_input_sizes.update({'future_prediction': self.hidden_size})
        self.pre_scalars = nn.ModuleDict(pre_scalar)

        self._device = torch.device('cpu')

        if not dataset_properties['uni_variant']:
            self.static_variable_selection = VariableSelectionNetwork(
                input_sizes=static_features_input_size,
                hidden_size=self.hidden_size,
                input_embedding_flags={},
                dropout=network_structure.grn_dropout_rate,
                prescalers=self.pre_scalars
            )
        self.static_input_sizes = static_features_input_size
        self.static_features = static_features

        self.auto_regressive = auto_regressive

        # create single variable grns that are shared across decoder and encoder
        if network_structure.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hidden_size),
                    self.hidden_size,
                    network_structure.grn_dropout_rate,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, self.hidden_size),
                        self.hidden_size,
                        network_structure.grn_dropout_rate,
                    )

        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},
            dropout=network_structure.grn_dropout_rate,
            context_size=self.hidden_size,
            single_variable_grns={}
            if not network_structure.share_single_variable_networks
            else self.shared_single_variable_grns,
            prescalers=self.pre_scalars,
        )

        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},
            dropout=network_structure.grn_dropout_rate,
            context_size=self.hidden_size,
            single_variable_grns={}
            if not network_structure.share_single_variable_networks
            else self.shared_single_variable_grns,
            prescalers=self.pre_scalars,
        )

        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=network_structure.grn_dropout_rate,
        )

        n_hidden_states = 0
        if network_encoder['block_1'].encoder_properties.has_hidden_states:
            n_hidden_states = network_encoder['block_1'].n_hidden_states

        static_context_initial_hidden = [GatedResidualNetwork(input_size=self.hidden_size,
                                                              hidden_size=self.hidden_size,
                                                              output_size=self.hidden_size,
                                                              dropout=network_structure.grn_dropout_rate,
                                                              ) for _ in range(n_hidden_states)]

        self.static_context_initial_hidden = nn.ModuleList(static_context_initial_hidden)
        self.cached_static_contex = None
        self.cached_static_embedding = None

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self.to(device)
        self._device = device

    def forward(self,
                x_past: Optional[Dict[str, torch.Tensor]],
                x_future: Optional[Dict[str, torch.Tensor]],
                x_static: Optional[Dict[str, torch.Tensor]],
                length_past: int = 0,
                length_future: int = 0,
                batch_size: int = 0,
                cache_static_contex: bool = False,
                use_cached_static_contex: bool = False,
                ):
        if x_past is None and x_future is None:
            raise ValueError('Either past input or future inputs need to be given!')
        if length_past == 0 and length_future == 0:
            raise ValueError("Either length_past or length_future must be given!")
        timesteps = length_past + length_future

        if not use_cached_static_contex:
            if len(self.static_input_sizes) > 0:
                static_embedding, _ = self.static_variable_selection(x_static)
            else:
                model_dtype = next(iter(x_past.values())).dtype if length_past > 0 else next(
                    iter(x_future.values())).dtype
                static_embedding = torch.zeros(
                    (batch_size, self.hidden_size), dtype=model_dtype, device=self.device
                )
                static_variable_selection = torch.zeros((batch_size, 0), dtype=model_dtype, device=self.device)

            static_context_variable_selection = self.static_context_variable_selection(static_embedding)[:, None]
            static_context_initial_hidden = tuple(init_hidden(static_embedding) for init_hidden in
                                                  self.static_context_initial_hidden)
            if cache_static_contex:
                self.cached_static_contex = static_context_variable_selection
                self.cached_static_embedding = static_embedding
        else:
            static_embedding = self.cached_static_embedding
            static_context_initial_hidden = None
            static_context_variable_selection = self.cached_static_contex
        static_context_variable_selection = static_context_variable_selection.expand(-1, timesteps, -1)
        if x_past is not None:
            embeddings_varying_encoder, _ = self.encoder_variable_selection(
                x_past,
                static_context_variable_selection[:, :length_past],
            )
        else:
            embeddings_varying_encoder = None
        if x_future is not None:
            embeddings_varying_decoder, _ = self.decoder_variable_selection(
                x_future,
                static_context_variable_selection[:, length_past:],
            )
        else:
            embeddings_varying_decoder = None
        return embeddings_varying_encoder, embeddings_varying_decoder, static_embedding, static_context_initial_hidden


class StackedEncoder(nn.Module):
    def __init__(self,
                 network_structure: NetworkStructure,
                 has_temporal_fusion: bool,
                 encoder_info: Dict[str, EncoderBlockInfo],
                 decoder_info: Dict[str, DecoderBlockInfo],
                 ):
        super().__init__()
        self.num_blocks = network_structure.num_blocks
        self.skip_connection = network_structure.skip_connection
        self.has_temporal_fusion = has_temporal_fusion

        self.encoder_output_type = [EncoderOutputForm.NoOutput] * self.num_blocks
        self.encoder_has_hidden_states = [False] * self.num_blocks
        len_cached_intermediate_states = self.num_blocks + 1 if self.has_temporal_fusion else self.num_blocks
        self.cached_intermediate_state = [torch.empty(0) for _ in range(len_cached_intermediate_states)]

        self.encoder_num_hidden_states = [0] * self.num_blocks
        encoder = nn.ModuleDict()
        for i, block_idx in enumerate(range(1, self.num_blocks + 1)):
            block_id = f'block_{block_idx}'
            encoder[block_id] = encoder_info[block_id].encoder
            if self.skip_connection:
                input_size = encoder_info[block_id].encoder_output_shape[-1]
                skip_size = encoder_info[block_id].encoder_input_shape[-1]
                if network_structure.skip_connection_type == 'add':
                    encoder[f'skip_connection_{block_idx}'] = AddLayer(input_size, skip_size)
                elif network_structure.skip_connection_type == 'gate_add_norm':
                    encoder[f'skip_connection_{block_idx}'] = GateAddNorm(input_size,
                                                                          hidden_size=input_size,
                                                                          skip_size=skip_size,
                                                                          dropout=network_structure.grn_dropout_rate)
            if block_id in decoder_info:
                if decoder_info[block_id].decoder_properties.recurrent:
                    if decoder_info[block_id].decoder_properties.has_hidden_states:
                        # RNN -> RNN
                        self.encoder_output_type[i] = EncoderOutputForm.HiddenStates
                    else:
                        # Transformer -> Transformer
                        self.encoder_output_type[i] = EncoderOutputForm.Sequence
                else:
                    # Deep AR
                    self.encoder_output_type[i] = EncoderOutputForm.SequenceLast
            if encoder_info[block_id].encoder_properties.has_hidden_states:
                self.encoder_has_hidden_states[i] = True
                self.encoder_num_hidden_states[i] = encoder_info[block_id].n_hidden_states
            else:
                self.encoder_has_hidden_states[i] = False
        self.encoder = encoder

    def forward(self,
                encoder_input: torch.Tensor,
                additional_input: List[Optional[torch.Tensor]],
                output_seq: bool = False,
                cache_intermediate_state: bool = False,
                incremental_update: bool = False) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        A forward pass through the encoder
        Args:
             encoder_input (torch.Tensor): encoder input
             additional_input (List[Optional[torch.Tensor]]) additional input to the encoder, e.g., inital hidden states
             output_seq (bool) if a sequence output is generated
             cache_intermediate_state (bool): if store the intermediate values
             incremental_update (bool): if an incremental update is applied, this is normally applied for auto-regressive
                model, however, ony deepAR requires encoder to do incremental update, thus the decoder only need to
                receive the last output of the encoder
        """
        encoder2decoder = []
        x = encoder_input
        for i, block_id in enumerate(range(1, self.num_blocks + 1)):
            output_seq_i = (output_seq or self.has_temporal_fusion or block_id < self.num_blocks)
            encoder_i = self.encoder[f'block_{block_id}']  # type: EncoderNetwork
            if self.encoder_has_hidden_states[i]:
                if incremental_update:
                    hx = self.cached_intermediate_state[i]
                    fx, hx = encoder_i(x, output_seq=False, hx=hx)
                else:
                    rnn_num_layers = encoder_i.config['num_layers']
                    hx = additional_input[i]
                    if hx is None:
                        fx, hx = encoder_i(x, output_seq=output_seq_i, hx=hx)
                    else:
                        if self.encoder_num_hidden_states[i] == 1:
                            fx, hx = encoder_i(x, output_seq=output_seq_i,
                                               hx=hx[0].expand((rnn_num_layers, -1, -1)).contiguous())
                        else:
                            hx = tuple(hx_i.expand(rnn_num_layers, -1, -1).contiguous() for hx_i in hx)
                            fx, hx = encoder_i(x, output_seq=output_seq_i, hx=hx)
            else:
                if incremental_update:
                    x_all = torch.cat([self.cached_intermediate_state[i], x], dim=1)
                    fx = encoder_i(x_all, output_seq=False)
                else:
                    fx = encoder_i(x, output_seq=output_seq_i)
            if self.skip_connection:
                if output_seq_i:
                    fx = self.encoder[f'skip_connection_{block_id}'](fx, x)
                else:
                    fx = self.encoder[f'skip_connection_{block_id}'](fx, x[:, -1:])

            if self.encoder_output_type[i] == EncoderOutputForm.HiddenStates:
                encoder2decoder.append(hx)
            elif self.encoder_output_type[i] == EncoderOutputForm.Sequence:
                encoder2decoder.append(fx)
            elif self.encoder_output_type[i] == EncoderOutputForm.SequenceLast:
                if output_seq or incremental_update:
                    encoder2decoder.append(fx)
                elif output_seq_i:
                    encoder2decoder.append(encoder_i.get_last_seq_value(fx).squeeze(1))
                else:
                    encoder2decoder.append(fx.squeeze(1))

            if cache_intermediate_state:
                if self.encoder_has_hidden_states[i]:
                    self.cached_intermediate_state[i] = hx
                else:
                    if incremental_update:
                        self.cached_intermediate_state[i] = x_all
                    else:
                        self.cached_intermediate_state[i] = x
                    # otherwise the decoder does not exist for this layer
            x = fx

        if self.has_temporal_fusion:
            if incremental_update:
                self.cached_intermediate_state[i + 1] = torch.cat([self.cached_intermediate_state[i + 1], x], dim=1)
            else:
                self.cached_intermediate_state[i + 1] = x
            return encoder2decoder, x
        else:
            return encoder2decoder, None


class StackedDecoder(nn.Module):
    def __init__(self,
                 network_structure: NetworkStructure,
                 encoder: nn.ModuleDict,
                 encoder_info: Dict[str, EncoderBlockInfo],
                 decoder_info: Dict[str, DecoderBlockInfo],
                 ):
        super().__init__()
        self.num_blocks = network_structure.num_blocks
        self.first_block = None
        self.skip_connection = network_structure.skip_connection

        self.decoder_has_hidden_states = []
        decoder = nn.ModuleDict()
        for i in range(1, self.num_blocks + 1):
            block_id = f'block_{i}'
            if block_id in decoder_info:
                self.first_block = i if self.first_block is None else self.first_block
                decoder[block_id] = decoder_info[block_id].decoder
                if decoder_info[block_id].decoder_properties.has_hidden_states:
                    self.decoder_has_hidden_states.append(True)
                else:
                    self.decoder_has_hidden_states.append(False)
                if self.skip_connection:
                    input_size_encoder = encoder_info[block_id].encoder_output_shape[-1]
                    skip_size_encoder = encoder_info[block_id].encoder_input_shape[-1]

                    input_size_decoder = decoder_info[block_id].decoder_output_shape[-1]
                    skip_size_decoder = decoder_info[block_id].decoder_input_shape[-1]
                    if skip_size_decoder > 0:
                        if input_size_encoder == input_size_decoder and skip_size_encoder == skip_size_decoder:
                            decoder[f'skip_connection_{i}'] = encoder[f'skip_connection_{i}']
                        else:
                            if network_structure.skip_connection_type == 'add':
                                decoder[f'skip_connection_{i}'] = AddLayer(input_size_decoder, skip_size_decoder)
                            elif network_structure.skip_connection_type == 'gate_add_norm':
                                decoder[f'skip_connection_{i}'] = GateAddNorm(input_size_decoder,
                                                                              hidden_size=input_size_decoder,
                                                                              skip_size=skip_size_decoder,
                                                                              dropout=network_structure.grn_dropout_rate)
        self.cached_intermediate_state = [torch.empty(0) for _ in range(self.num_blocks + 1 - self.first_block)]
        self.decoder = decoder

    def forward(self,
                x_future: Optional[torch.Tensor],
                encoder_output: List[torch.Tensor],
                pos_idx: Optional[Tuple[int]] = None,
                cache_intermediate_state: bool = False,
                incremental_update: bool = False
                ) -> torch.Tensor:
        x = x_future
        for i, block_id in enumerate(range(self.first_block, self.num_blocks + 1)):
            decoder_i = self.decoder[f'block_{block_id}']  # type: DecoderNetwork
            if self.decoder_has_hidden_states[i]:
                if incremental_update:
                    hx = self.cached_intermediate_state[i]
                    fx, hx = decoder_i(x_future=x, encoder_output=hx, pos_idx=pos_idx)
                else:
                    fx, hx = decoder_i(x_future=x, encoder_output=encoder_output[i], pos_idx=pos_idx)
            else:
                if incremental_update:
                    # in this case, we only have Transformer, thus x_all needs to be None value!
                    # TODO make this argument clearer!
                    #x_all = torch.cat([self.cached_intermediate_state[i], x], dim=1)
                    fx = decoder_i(x, encoder_output=encoder_output[i], pos_idx=pos_idx)
                else:
                    fx = decoder_i(x, encoder_output=encoder_output[i], pos_idx=pos_idx)
            skip_id = f'skip_connection_{block_id}'
            if self.skip_connection and skip_id in self.decoder and x is not None:
                fx = self.decoder[skip_id](fx, x)
            if cache_intermediate_state:
                if self.decoder_has_hidden_states[i]:
                    self.cached_intermediate_state[i] = hx
                    #TODO consider if there are other case that could make use of cached intermediate states
                # else:
                #    if incremental_update:
                #        self.cached_intermediate_state[i] = x_all
                #    else:
                #        self.cached_intermediate_state[i] = x
            x = fx
        return x