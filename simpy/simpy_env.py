"""OpenAI Environment Wrapper for a SimpyModel."""

from gym.spaces import Discrete
from gym import Env as GymEnv


class SimpyEnv(GymEnv):

    def __init__(self, config: dict):
        self.action_space = Discrete(config["n_actions"])
        self.observation_space = config["observation_space"]
        self.sim_model = config["sim_model"]
        self.sim_config = config["sim_config"]

    def reset(self):
        self.sim = self.sim_model(config=self.sim_config)

        # Start processes and initialize resources
        obs = self.sim.get_observation()
        assert self.observation_space.contains(obs), "{} not in {}".format(obs, self.observation_space)
        return obs

    def step(self, action):
        assert action in range(self.action_space.n)

        self.sim.exec_action(action)
        obs = self.sim.get_observation()
        reward, done, info = self.sim.get_reward()

        assert self.observation_space.contains(obs), "{} not in {}".format(obs, self.observation_space)
        return obs, reward, done, info
