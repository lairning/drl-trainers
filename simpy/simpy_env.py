"""OpenAI Environment Wrapper for a SimpyModel."""

import gym
from gym.spaces import Space, Discrete

class SimpyEnv(gym.Env):

    def __init__(self, n_actions: int, observation_space: Space, sim_model):
        self.action_space = Discrete(n_actions)
        self.observation_space = observation_space
        self.sim_model = sim_model

    def reset(self):
        self.sim = self.sim_model()

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



'''
### Unfortunately deepcopy does not work with yeld
from copy import deepcopy

class SimAlphaEnv:

    def __init__(self):
        self.env = SimpyEnv()
        self.action_space = Discrete(N_ACTIONS)
        self.observation_space = Dict({
            "obs"        : OBSERVATION_SPACE,
            "action_mask": Box(low=0, high=1, shape=(self.action_space.n,))
        })

    def reset(self):
        obs = self.env.reset()
        action_mask = np.array([1, 1-obs[3]])
        return {'obs': obs, "action_mask": action_mask}

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        if done:
            reward = self.env.sim.actual_revenue
        else:
            reward = 0
        action_mask = np.array([1, 1-obs[3]])
        obs = {'obs': obs, "action_mask": action_mask}
        return obs, reward, done, info

    def set_state(self, state):
        self.env = deepcopy(state[0])
        obs = self.env.sim.get_observation()
        action_mask = np.array([1, 1 - obs[3]])
        return {'obs': obs, "action_mask": action_mask}

    def get_state(self):
        return deepcopy(self.env), None

'''
