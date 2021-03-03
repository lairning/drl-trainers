import itertools

import numpy as np
import simpy
from gym.spaces import Box
import random

# Mandatory
BASE_CONFIG = {
    "SIM_HOURS": 2, # Simulation time in hours
    "ACTION_INTERVAL_SECONDS": 20, # Time between each action can be performed
    "MTBC": [40, 30, 50, 60, 40, 20, 70, 60] # Mean Time Between Cars Factor (lower increases frequency)
}

class BaseSim(simpy.Environment):
    def __init__(self, config:dict=None):
        super().__init__()
        self.sim_config = BASE_CONFIG.copy()
        if config is not None:
            self.sim_config.update(config)
        self.time = self.now
        self.sim_time = config["SIM_HOURS"] * 60 * 60 # Simulation time in Time units (seconds)
        self.step_time = config["ACTION_INTERVAL_SECONDS"]  # Time Units to run between gym env steps

    def run_until_action(self):
        self.run(until=self.time + self.step_time)
        self.time = self.now

    def done(self):
        return self.now >= self.sim_time

    def exec_action(self, action):
        raise Exception("Not Implemented!!!")

    def get_observation(self):
        raise Exception("Not Implemented!!!")

    def get_reward(self):
        raise Exception("Not Implemented!!!")


DEBUG = False

def dprint(*args):
    if DEBUG:
        print(*args)


def hot_encode(n, N):
    encode = [0] * N
    encode[n] = 1
    return encode

### Implement from Here

LIGHTS = ['South/North', 'North/South', 'South/West', 'North/East', 'West/East', 'East/West', 'West/North',
          'East/South']

# List of possible status, 1 Green On; 0 Green Off
STATUS_N = [
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1]
]
STATUS = [[bool(n) for n in status] for status in STATUS_N]

MAX_WAITING_TIME = np.inf # 400.0

class Light(simpy.PriorityResource):
    def __init__(self, name: str, env: simpy.Environment, green_on: bool, mtbc: float):
        super().__init__(env, capacity=1)
        self.name = name
        self.env = env
        self.green_on = True
        self.turn_green = None
        self.stats = {'waiting_time': []}
        self.queue = {}
        env.process(self.set_status(green_on))
        env.process(self.car_generator(mtbc))

    def set_status(self, green_on: bool):
        if self.green_on != green_on:
            self.green_on = not self.green_on
            if not green_on:
                # Turned Red
                dprint('{} turned RED at {}.'.format(self.name, self.env.now))
                self.turn_green = self.env.event()
                with self.request(priority=0) as req:
                    yield req
                    yield self.turn_green
                    yield self.env.timeout(10)
            else:
                # Turned Green
                dprint('{} turned Green at {}.'.format(self.name, self.env.now))
                self.turn_green.succeed()

    def car_generator(self, mtbc: float):
        """Generate new cars."""
        for i in itertools.count():
            yield self.env.timeout(random.expovariate(1 / mtbc))
            self.env.process(self.car_crossing(i))

    def car_crossing(self, n: int):
        with self.request(priority=1) as req:
            arrive_time = self.env.now

            queued = False if self.green_on else True
            if queued: self.queue[n] = arrive_time

            # Request access
            yield req

            if queued: del self.queue[n]

            # waiting time
            self.stats['waiting_time'].append(self.env.now - arrive_time)

            yield self.env.timeout(3)
            dprint('Car {} waited {:.2f} minutes on {}.'.format(n, (self.env.now - arrive_time) / 60, self.name))

    def get_observation(self):
        waiting_time_sum = sum((self.env.now-value) for value in self.queue.values())
        return [1 if self.green_on else 0]+[min(waiting_time_sum, MAX_WAITING_TIME)]

# An action corresponds to the selection of a status
N_ACTIONS = len(STATUS)

OBSERVATION_SPACE = Box(low=np.array([0,0]*len(LIGHTS)),
                        high=np.array([1, MAX_WAITING_TIME] * len(LIGHTS)),
                        dtype=np.float64)

class SimModel(BaseSim):
    def __init__(self, config:dict=None):
        super().__init__(config)
        mtbc = self.sim_config['MTBC']
        self.lights = [Light(LIGHTS[i], self, STATUS[0][i], mtbc[i]) for i in range(len(LIGHTS))]
        self.total_reward = 0

    def get_observation(self):
        self.run_until_action()
        obs = []
        for light in self.lights:
            obs += light.get_observation()
        return obs

    def get_reward(self):
        q_value = [[self.now-value for value in light.queue.values()] for light in self.lights ]
        total_reward = - max([max(lq) if len(lq) else 0 for lq in q_value])
        reward = total_reward - self.total_reward
        self.total_reward = total_reward
        return reward, self.done(), {}  # Reward, Done, Info

    # Executes an action
    def exec_action(self, action):
        # An action set the status of the lights according to the STATUS Table
        for i, light in enumerate(self.lights):
            self.process(light.set_status(STATUS[action][i]))
        self.current_status_id = action

# Mandatory
class SimBaseline:
    def __init__(self, baseline_config:dict=None, sim_config:dict=None):
        self.baseline_config = {"round_robin": 2}
        self.sim = None
        if baseline_config is not None:
            self.baseline_config.update(baseline_config)
        self.sim_config = BASE_CONFIG.copy()
        if sim_config is not None:
            self.sim_config.update(sim_config)

    class RandomAction:
        def get(self):
            return random.choice(list(range(N_ACTIONS)))

    class RoundRobin:
        def __init__(self, interval):
            self.i = 0
            self.interval = interval
            self.j = 0

        def get(self):
            self.j += 1
            if self.j == self.interval:
                self.j = 0
                self.i += 1
                if self.i == N_ACTIONS:
                    self.i = 0
            return self.i

    def run(self):
        self.sim = SimModel(self.sim_config)
        policy = self.RoundRobin(self.baseline_config["round_robin"])
        done = False
        total_reward = 0
        while not done:
            obs = self.sim.get_observation()
            action = policy.get()
            self.sim.exec_action(action)
            reward, done, _ = self.sim.get_reward()
            total_reward += reward

        return total_reward

def print_stats(sim: SimModel):
    total_cars = 0
    total_waiting_time = 0
    for light in sim.lights:
        cars = len(light.stats['waiting_time'])
        waiting_time = sum(light.stats['waiting_time'])
        cars_q = len([(sim.now-value) for value in light.queue.values()])
        waiting_time_q = sum([(sim.now-value) for value in light.queue.values()])
        total_cars += cars + cars_q
        total_waiting_time += waiting_time + waiting_time_q
        avg_time = 0 if cars==0 else waiting_time/cars
        avg_time_queue = 0 if cars_q==0 else waiting_time_q/cars_q
        print("{} - Total Cars: {}; Average Waiting Time: {:.2f}; {} Cars Stopped with Average Waiting Time: {:.2f}".
              format(light.name, cars, avg_time, cars_q, avg_time_queue))
    print("### Total Cars: {}; Average waiting: {:.2f}".format(total_cars, total_waiting_time / total_cars))
    #print(len([1 for x in light.stats['waiting_time'] if x==0]))

if __name__ == "__main__":

    import ray
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default=1, type=int)
    parser.add_argument("--inst", default="auto", type=str)

    args = parser.parse_args()

    # @ray.remote
    def base_run(baseline):
        return baseline.run()

    # ray.init(address="auto")

    total = 0
    baseline = SimBaseline()
    # results = ray.get([base_run.remote(baseline) for _ in range(args.n)])
    results = [base_run(baseline) for _ in range(args.n)]
    print("### Average Rewards", np.mean(results))

    ray.shutdown()