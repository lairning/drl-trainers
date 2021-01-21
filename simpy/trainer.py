import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import ActorCriticModel
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.preprocessors import get_preprocessor

torch, nn = try_import_torch()

import argparse

from simpy_model import SimpyEnv, SimAlphaEnv

class DenseModel(ActorCriticModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        #print("################# obs_space:",obs_space)
        '''
        ActorCriticModel.__init__(self, obs_space, action_space, num_outputs,
                                  model_config, name)
        '''
        self.preprocessor = get_preprocessor(obs_space)(obs_space)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        self.shared_layers = nn.Sequential(
            nn.Linear(
                in_features=obs_space.original_space["obs"].shape[0],
                out_features=256), nn.Linear(
                    in_features=256, out_features=256))
        self.actor_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=action_space.n))
        self.critic_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=1))
        self._value_out = None

parser = argparse.ArgumentParser()
parser.add_argument("--stop", type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init()

    register_env(
        "SimpyEnv",
        lambda _: SimpyEnv()
    )

    register_env(
        "SimAlphaEnv",
        lambda _: SimAlphaEnv()
    )

    ModelCatalog.register_custom_model("dense_model", DenseModel)

    dqn_config = {
        "v_min"      : -30000.0,
        "v_max"      : 1000.0,
        "env"        : "SimpyEnv",
        "hiddens"    : [256, 256],
        "num_workers": 5
    }

    ppo_config = {
        "env"          : "SimpyEnv",
        "vf_clip_param": 50,  # tune.grid_search([20.0, 100.0]),
        "num_workers"  : 5,
        # "lr"            : tune.grid_search([1e-4, 1e-6]),
        "batch_mode"   : "complete_episodes"
    }

    alpha_config = {
        "env"           : "SimAlphaEnv",
        "mcts_config"   : {
            "puct_coefficient"   : 1.0,
            "num_simulations"    : 30,
            "temperature"        : 1.5,
            "dirichlet_epsilon"  : 0.25,
            "dirichlet_noise"    : 0.03,
            "argmax_tree_policy" : False,
            "add_dirichlet_noise": True,
        },
        "ranked_rewards": {
            "enable": True,
        },
        "model": {
            "custom_model": "dense_model",
        },
    }

    stop = {
        "training_iteration": args.stop
    }

    results_alpha = tune.run("contrib/AlphaZero", stop=stop, max_failures=0, config=alpha_config)

    # results_ppo = tune.run(ppo.PPOTrainer, config=ppo_config, stop=stop)

    ray.shutdown()
