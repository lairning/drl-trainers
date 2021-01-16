
import numpy as np
import gym
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.misc import SlimFC, normc_initializer, AppendBiasLayer

torch, nn = try_import_torch()

CARD_EMBEDD_SIZE = 3

FIRST_LAYER_SIZE = 3 * 4 + CARD_EMBEDD_SIZE * 4

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

        self._embedd = nn.Embedding(int(obs_space.high[-1]) + 1, CARD_EMBEDD_SIZE)

        # Player Hot Encoded = 3 * Number of Cards Played per trick = 4
        # CARD_EMBEDD_SIZE * Number of Cards Played per trick = 4

        self._hidden_layers = self._build_hidden_layers(first_layer_size=FIRST_LAYER_SIZE,
                                                        hiddens=hiddens,
                                                        activation=activation)

        self._value_branch_separate = None
        self._value_embedding = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            self._value_embedding = nn.Embedding(int(obs_space.high[-1]) + 1, CARD_EMBEDD_SIZE)
            self._value_branch_separate = self._build_hidden_layers(first_layer_size=FIRST_LAYER_SIZE,
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
        self._players_in, self._cards_in = torch.split(input_dict['obs_flat'],[12,4],1)
        self._cards_in = self._cards_in.long()
        emb_cards = self._embedd(self._cards_in).reshape(self._cards_in.shape[0],-1)
        obs_flat = torch.cat((self._players_in, emb_cards), 1)
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
            emb_cards = self._value_embedding(self._cards_in).reshape(self._cards_in.shape[0],-1)
            obs_flat = torch.cat((self._players_in, emb_cards), 1)
            return self._value_branch(
                self._value_branch_separate(obs_flat.reshape(obs_flat.shape[0], -1))).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)


from ray.rllib.contrib.alpha_zero.models.custom_torch_models import ActorCriticModel, convert_to_tensor
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX

class AlphaHeartsModel(ActorCriticModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        ActorCriticModel.__init__(self, obs_space, action_space, num_outputs,
                                  model_config, name)

        self._embedd = nn.Embedding(52 + 1, CARD_EMBEDD_SIZE)

        #print("## DEBUG obs_space.original_space ###", obs_space.original_space)
        N_NEURONS = 256
        self.shared_layers = nn.Sequential(
            nn.Linear(
                in_features= FIRST_LAYER_SIZE,
                out_features=N_NEURONS),
            nn.Linear(in_features=N_NEURONS, out_features=N_NEURONS)
        )
        self.actor_layers = nn.Sequential(
            nn.Linear(in_features=N_NEURONS, out_features=action_space.n))
        self.critic_layers = nn.Sequential(
            nn.Linear(in_features=N_NEURONS, out_features=1))
        self._value_out = None

    def compute_priors_and_value(self, obs):
        obs = convert_to_tensor([self.preprocessor.transform(obs)])
        input_dict = restore_original_dimensions(obs, self.obs_space, "torch")
        with torch.no_grad():
            model_out = self.forward(input_dict, None, [1])
            logits, _ = model_out
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()

            return priors, value

    def forward(self, input_dict, state, seq_lens):

        action_mask = input_dict["action_mask"]

        self._players_in, self._cards_in = torch.split(input_dict['obs'],[12,4],1)
        self._cards_in = self._cards_in.long()
        emb_cards = self._embedd(self._cards_in).reshape(self._cards_in.shape[0],-1)
        x = torch.cat((self._players_in, emb_cards), 1)

        x = self.shared_layers(x)
        # actor outputs
        logits = self.actor_layers(x)

        # compute value
        self._value_out = self.critic_layers(x)

        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        print("##############  DEBUG")
        print(logits)
        print(inf_mask)
        print(logits + inf_mask)
        return logits + inf_mask, None

