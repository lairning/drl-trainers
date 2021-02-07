import ray
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
from gym.spaces import Space
import simpy

from simpy_env import SimpyEnv2

class Trainer():
    ppo_config = {
        "vf_clip_param": 10,  # tune.grid_search([20.0, 100.0]),
        "num_workers"  : 5,
        # "lr"            : tune.grid_search([1e-4, 1e-6]),
        "batch_mode"   : "complete_episodes",
        "framework"    : "torch"
    }

    def __init__(self, n_actions: int, observation_space: Space, sim_model: simpy.Environment, trainer_config):
        _config = ppo_config.copy()
        _config.update(trainer_config)
        _config["env"] = SimpyEnv2
        _config["env_config"] = {"n_actions" : n_actions,
                                 "observation_space" : observation_space,
                                 "sim_model" : sim_model}
        self._trainer = ppo.PPOTrainer(config=_config, env="SimpyEnv")

    def run(self, sessions: int):

        ray.init()

        result = self._trainer.train()
        best_checkpoint = self._trainer.save()
        best_reward = result['episode_reward_mean']
        print("Mean Reward {}:{}".format(1, result['episode_reward_mean']))

        for i in range(1, sessions):
            result = self._trainer.train()
            print("Mean Reward {}:{}".format(i + 1, result['episode_reward_mean']))
            best_reward = max(best_reward, result['episode_reward_mean'])
            if best_reward == result['episode_reward_mean']:
                best_checkpoint = self._trainer.save()

        print("BEST Mean Reward  :", best_reward)
        print("BEST Checkpoint at:", best_checkpoint)

        ray.shutdown()


if __name__ == "__main__":

    from trafic_light_model import N_ACTIONS, OBSERVATION_SPACE, SimModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", type=int, default=1)

    args = parser.parse_args()

    trainer = Trainer(N_ACTIONS, OBSERVATION_SPACE, SimModel)
    trainer.run(2)
