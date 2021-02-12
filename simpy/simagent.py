import ray
import ray.rllib.agents.ppo as ppo
from gym.spaces import Space

from simpy_env import SimpyEnv2

def get_train_result(result: dict):
    dic = result.copy()
    dic.pop("config", None)
    dic.pop("hist_stats", None)
    return dic


class AISimAgent():
    ppo_config = {
        "vf_clip_param": 10,  # tune.grid_search([20.0, 100.0]),
        "num_workers"  : 5,
        # "lr"            : tune.grid_search([1e-4, 1e-6]),
        "batch_mode"   : "complete_episodes",
        "framework"    : "torch",
        "log_level"    : "ERROR"
    }

    def __init__(self, n_actions: int, observation_space: Space, sim_model, agent_config=None):
        if agent_config is None:
            agent_config = {}
        else:
            assert isinstance(agent_config, dict), "Config {} must be a dict!".format(agent_config)
        self._config = self.ppo_config.copy()
        self._config.update(agent_config)
        self._config["env"] = SimpyEnv2
        self._config["env_config"] = {"n_actions" : n_actions,
                                 "observation_space" : observation_space,
                                 "sim_model" : sim_model}
        self.runs = None

    def train(self, sessions: int = 1, config=None, log: bool = False):

        if config is None:
            config = dict()
        else:
            assert isinstance(config, dict), "Config {} must be a dict!".format(config)

        ray.init()

        self._trainer = ppo.PPOTrainer(config=self._config)

        result_list = []
        result = self._trainer.train()
        print(get_train_result(result))
        best_checkpoint = self._trainer.save()
        best_reward = result['episode_reward_mean']
        if log: print("Mean Reward {}:{}".format(1, result['episode_reward_mean']))
        result_list.append(result['episode_reward_mean'])

        for i in range(1, sessions):
            result = self._trainer.train()
            if log: print("Mean Reward {}:{}".format(i + 1, result['episode_reward_mean']))
            result_list.append(result['episode_reward_mean'])
            best_reward = max(best_reward, result['episode_reward_mean'])
            if best_reward == result['episode_reward_mean']:
                best_checkpoint = self._trainer.save()

        if log: print("BEST Mean Reward  :", best_reward)
        if log: print("BEST Checkpoint at:", best_checkpoint)

        ray.shutdown()

        return best_checkpoint, result_list

if __name__ == "__main__":

    from trafic_light_model import N_ACTIONS, OBSERVATION_SPACE, SimModel
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", type=int, default=2)

    args = parser.parse_args()

    trainer = AISimAgent(N_ACTIONS, OBSERVATION_SPACE, SimModel, {})
    trainer.train(args.stop)
