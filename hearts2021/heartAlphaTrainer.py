import ray
from ray import tune
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import ActorCriticModel
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env
from env import HeartsAlphaEnv

from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

class DenseModel(ActorCriticModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        ActorCriticModel.__init__(self, obs_space, action_space, num_outputs,
                                  model_config, name)

        #print("## DEBUG obs_space ###", obs_space)
        self.shared_layers = nn.Sequential(
            nn.Linear(
                in_features= 20, #obs_space.original_space["obs"].shape[0],
                out_features=256), nn.Linear(
                    in_features=256, out_features=256))
        self.actor_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=action_space.n))
        self.critic_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=1))
        self._value_out = None

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
        stop={"training_iteration": 10000},
        max_failures=0,
        #resources_per_trial={"cpu": 2, "extra_cpu":2},
        config={
            "env": "HeartsEnv",
            "num_workers": 5,
            "rollout_fragment_length": 50,
            "train_batch_size": 500,
            "sgd_minibatch_size": 64,
            "lr": 1e-4,
            "num_sgd_iter": 1,
            "mcts_config": {
                "puct_coefficient": 1.5,
                "num_simulations": 100,
                "temperature": 1.0,
                "dirichlet_epsilon": 0.20,
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