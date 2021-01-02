import ray
from ray import tune
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import ActorCriticModel
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env
from env import HeartsAlphaEnv

from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX

torch, nn = try_import_torch()

class DenseModel(ActorCriticModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        ActorCriticModel.__init__(self, obs_space, action_space, num_outputs,
                                  model_config, name)

        #print("## DEBUG obs_space.original_space ###", obs_space.original_space)
        self.shared_layers = nn.Sequential(
            nn.Linear(
                in_features= obs_space.original_space["obs"].n,
                out_features=256), nn.Linear(
                    in_features=256, out_features=256))
        self.actor_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=action_space.n))
        self.critic_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=1))
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):

        action_mask = input_dict["action_mask"]

        x = input_dict["obs"]
        x = self.shared_layers(x)
        # actor outputs
        logits = self.actor_layers(x)

        # compute value
        self._value_out = self.critic_layers(x)

        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        return logits + inf_mask, None

if __name__ == "__main__":

    ray.init()

    ModelCatalog.register_custom_model("dense_model", DenseModel)

    register_env(
        "HeartsEnv",
        #lambda _: HeartsEnv()
        lambda _: HeartsAlphaEnv(10)
    )

    tune.run(
        "contrib/AlphaZero",
        stop={"training_iteration": 2},
        max_failures=0,
        #resources_per_trial={"cpu": 2, "extra_cpu":2},
        config={
            "env": "HeartsEnv",
            "num_workers": 0,
            "rollout_fragment_length": 200,
            "train_batch_size": 4000,
            "sgd_minibatch_size": 128,
            "lr": 1e-4,
            "num_sgd_iter": 30,
            "mcts_config": {
                "puct_coefficient": 1.0,
                "num_simulations": 30,
                "temperature": 1.5,
                "dirichlet_epsilon": 0.25,
                "dirichlet_noise": 0.03,
                "argmax_tree_policy": False,
                "add_dirichlet_noise": True,
            },
            "ranked_rewards": {
                "enable": True,
            },
            "model": {
                "custom_model": "dense_model",
            },
        },
    )