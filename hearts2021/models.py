
import numpy as np
import gym
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.misc import SlimFC, normc_initializer, AppendBiasLayer

torch, nn = try_import_torch()

CARD_EMBEDD_SIZE = 3

class HeartsNetwork(TorchModelV2, nn.Module):
    """Customized PPO network."""

    def _build_hidden_layers(self, first_layer_size: int, hiddens: list, activation: str):

        layers = []

        prev_layer_size = first_layer_size

        # Create layers. Assumes no_final_linear = False
        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            prev_layer_size = size

        return nn.Sequential(*layers)


    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        # Nonlinearity for fully connected net (tanh, relu). Default: "tanh"
        activation = model_config.get("fcnet_activation")
        # Number of hidden layers for fully connected net. Default: [256, 256]
        hiddens = [256,256] # model_config.get("fcnet_hiddens", [])
        # Whether to skip the final linear layer used to resize the hidden layer
        # outputs to size `num_outputs`. If True, then the last hidden layer
        # should already match num_outputs.
        # no_final_linear = False

        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = False

        self._embedd = nn.Embedding(int(obs_space['cards'].high[0]) + 1, CARD_EMBEDD_SIZE)

        # Player Hot Encoded = 3 * Number of Cards Played per trick = 4
        # CARD_EMBEDD_SIZE * Number of Cards Played per trick = 4
        first_layer_size = 3*4+CARD_EMBEDD_SIZE*4
        self._hidden_layers = self._build_hidden_layers(first_layer_size=first_layer_size,
                                                        hiddens=hiddens,
                                                        activation=activation)

        self._value_branch_separate = None
        self._value_embedding = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            self._value_embedding = nn.Embedding(int(obs_space['cards'].high[0]) + 1, CARD_EMBEDD_SIZE)
            self._value_branch_separate = self._build_hidden_layers(first_layer_size=first_layer_size,
                                                                    hiddens=hiddens,
                                                                    activation=activation)
        self._logits = SlimFC(
            in_size=hiddens[-1],
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        self._value_branch = SlimFC(
            in_size=hiddens[-1],
            out_size=1,
            initializer=normc_initializer(1.0),
            activation_fn=None)
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._cards_in = None
        self._players_in = None

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        self._cards_in = torch.LongTensor(input_dict['obs']['cards'])
        self._players_in = torch.LongTensor(input_dict['obs']['players'])
        emb_cards = self._embedd(self._cards_in)
        obs_flat = torch.cat((self._players_in, emb_cards), 1)
        print("#######   DEBUG   ######:",input_dict.shape, obs_flat.shape)
        self._features = self._hidden_layers(obs_flat.reshape(obs_flat.shape[0], -1))
        logits = self._logits(self._features) if self._logits else \
            self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            emb_cards = self._value_embedding(self._cards_in)
            obs_flat = torch.cat((self._players_in, emb_cards), 1)
            return self._value_branch(
                self._value_branch_separate(obs_flat.reshape(obs_flat.shape[0], -1))).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)

class FullyConnectedNetwork(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens", [])
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            prev_layer_size = size


        print("### DEBUG ####",no_final_linear)
        print("### DEBUG ####",num_outputs)
        print("### DEBUG ####",hiddens)
        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if num_outputs:
            self._logits = SlimFC(
                in_size=prev_layer_size,
                out_size=num_outputs,
                initializer=normc_initializer(0.01),
                activation_fn=None)

        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            prev_vf_layer_size = int(np.product(obs_space.shape))
            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0)))
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(1.0),
            activation_fn=None)
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else \
            self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            return self._value_branch(
                self._value_branch_separate(self._last_flat_in)).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)