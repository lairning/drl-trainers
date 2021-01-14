"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import numpy as np
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
import ray.rllib.agents.ppo as ppo
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX

from env import HeartsParametricEnv1, TRUE_OBSERVATION_SPACE1
from models import HeartsNetwork

torch, nn = try_import_torch()

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument("--run", type=str, default="PPO")
#parser.add_argument("--torch", action="store_true")
#parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=50)
parser.add_argument("--workers", type=int, default=5)
parser.add_argument("--stop-reward", type=float, default=4)


class TorchParametricActionsModel(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kw):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs,
                               model_config, name, **kw)

        self.action_model = HeartsNetwork(
            TRUE_OBSERVATION_SPACE1, action_space, num_outputs,
            model_config, name + "_custom_hearts")

        #print(self.action_model)


    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # print(input_dict["obs"])
        print("##### DEBUG #####",input_dict["obs"]["obs"])

        # Compute the predicted action embedding
        action_param, _ = self.action_model({
            "obs": np.array(input_dict["obs"]["obs"])
        })

        # Mask out invalid actions (use -inf to tag invalid).
        # These are then recognized by the EpsilonGreedy exploration component
        # as invalid actions that are not to be chosen.
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        return action_param + inf_mask, state

    def value_function(self):
        return self.action_model.value_function()


if __name__ == "__main__":

    args = parser.parse_args()
    ray.init()

    ModelCatalog.register_custom_model(
        "my_model", TorchParametricActionsModel)

    register_env(
        "HeartsEnv",
        lambda _: HeartsParametricEnv1(random_players=True)
    )

    config_default = {
        "env": "HeartsEnv",
        "model": {
            "custom_model": "my_model",
        },
        "vf_share_layers": False,
        "framework": "torch" # if args.torch else "tf",
    }

    config_tuned = {
        "env": "HeartsEnv",
        "model": {
            "custom_model": "my_model",
        },
        "vf_share_layers": False,
        #"lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        "framework": "torch" # if args.torch else "tf",
    }

    stop = {
        "training_iteration": args.stop_iters,
        #"episode_reward_mean": args.stop_reward,
    }

    config = config_default
    config["num_workers"] = args.workers

    results = tune.run("PPO", config=config, stop=stop, checkpoint_at_end=True)
    # results = tune.run("DQN", config=config_dqn, stop=stop)

    best_checkpoint = results.get_best_checkpoint(trial=results.get_best_trial(metric="episode_reward_mean",
                                                                               mode="max"),
                                                  metric="episode_reward_mean",
                                                  mode="max")

    print(best_checkpoint)

    config["num_workers"] = 0

    agent = ppo.PPOTrainer(config=config, env="HeartsEnv")
    agent.restore(best_checkpoint)

    # instantiate env class
    he = HeartsParametricEnv1(random_players=True)

    # run until episode ends
    episode_reward = 0
    done = False
    obs = he.reset()
    while not done:
        #print(obs)
        action = agent.compute_action(obs)
        print(he.env.me, he.env.table_card, he._decode_card(action))
        obs, reward, done, info = he.step(action)
        episode_reward += reward
        print(episode_reward,reward)

    ray.shutdown()

