import itertools

import numpy as np
import simpy
from gym.spaces import Box
import random

# SIM_TIME = 1 * 24 * 60 * 60  # Simulation time in Time units (seconds)
SIM_TIME = 1 * 1 * 60 * 60  # Simulation time in Time units (seconds)
STEP_TIME = 20  # Time units (seconds) between each step


class BaseSim(simpy.Environment):
    def __init__(self):
        super().__init__()
        self.time = self.now
        self.sim_time = SIM_TIME
        self.step_time = STEP_TIME  # Time Units to run between gym env steps

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


# LIGHTS = ['sn','ns','sw','ne','we','ew','wn','es']
LIGHTS = ['South/North', 'North/South', 'South/West', 'North/East', 'West/East', 'East/West', 'West/North',
          'East/South']
MTBC_BASE = [45, 60, 30, 40, 40, 70, 20, 30]
# Mean Time Between Cars
MTBC = [x * 0.4 for x in MTBC_BASE]

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

MAX_QUEUE = np.inf # 400.0

class Light(simpy.PriorityResource):
    def __init__(self, name: str, env: simpy.Environment, green_on: bool, mtbc: float):
        super().__init__(env, capacity=1)
        self.name = name
        self.env = env
        self.green_on = True
        self.turn_green = None
        self.stats = {'total_cars': 0, 'waiting_time': 0.0}
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
            # Request access
            yield req

            # waiting time
            self.stats['waiting_time'] += (self.env.now - arrive_time)

            yield self.env.timeout(3)
            dprint('Car {} waited {:.2f} minutes on {}.'.format(n, (self.env.now - arrive_time) / 60, self.name))
            self.stats['total_cars'] += 1

    def get_observation(self):
        return [1 if self.green_on else 0]+[min(len(self.queue),MAX_QUEUE)]

# An action corresponds to the selection of a status
N_ACTIONS = len(STATUS)

OBSERVATION_SPACE = Box(low=np.array([0,0]*len(LIGHTS)),
                        high=np.array([1, MAX_QUEUE]*len(LIGHTS)),
                        dtype=np.float64)

class SimModel(BaseSim):
    def __init__(self):
        super().__init__()
        #self.current_status_id = 0
        self.lights = [Light(LIGHTS[i], self, STATUS[0][i], MTBC[i]) for i in range(len(LIGHTS))]
        self.total_reward = 0

    def get_observation(self):
        self.run_until_action()
        obs = []
        for light in self.lights:
            obs += light.get_observation()
        return obs

    def get_reward(self):
        def qwt(light:Light):
            return (self.now if light.stats['total_cars']==0
                    else light.stats['waiting_time'] / light.stats['total_cars'])* len(light.queue)
        total_cars = sum(light.stats['total_cars'] for light in self.lights)
        waiting_time = sum(light.stats['waiting_time'] for light in self.lights)
        total_cars += sum(len(light.queue) for light in self.lights)
        waiting_time += sum(qwt(light) for light in self.lights)
        total_reward = 0 if total_cars == 0 else -waiting_time/total_cars
        reward = total_reward - self.total_reward
        self.total_reward = total_reward
        return reward, self.done(), {}  # Reward, Done, Info

    # Executes an action
    def exec_action(self, action):
        # An action set the status of the lights according to the STATUS Table
        for i, light in enumerate(self.lights):
            self.process(light.set_status(STATUS[action][i]))
        self.current_status_id = action


class Baseline:
    def __init__(self):
        self.sim = SimModel()

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

    def run(self, policy):
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
    l_waiting_time = []
    for light in sim.lights:
        waiting_time = 0 if light.stats['total_cars'] == 0 else light.stats['waiting_time'] / light.stats['total_cars']
        l_waiting_time += [waiting_time]
        print("{} - Total Cars: {}; Average Waiting Time: {:.2f}; {} Cars Stopped".
              format(light.name, light.stats['total_cars'], waiting_time, len(light.queue)))
    total_cars = sum(light.stats['total_cars'] for light in sim.lights)
    waiting_time = sum(light.stats['waiting_time'] for light in sim.lights)
    print("### Total Cars: {}; Average waiting: {:.2f}".format(total_cars, waiting_time / total_cars))

    q_total_cars = sum(len(light.queue) for light in sim.lights)
    q_estimated_time = sum(light.stats['waiting_time'] * len(light.queue) / light.stats['total_cars'] for i, light in
                           enumerate(sim.lights))

    total_cars += q_total_cars
    w_waiting_time = sum(light.stats['waiting_time'] for light in sim.lights) + q_estimated_time
    print("### Reward: {:.2f}".format(-w_waiting_time / total_cars))
    print("### STD: {:.2f}".format(np.std(l_waiting_time)))


if __name__ == "__main__":
    n = 20
    total = 0
    for _ in range(n):
        baseline = Baseline()
        policy = baseline.RoundRobin(10)
        reward = baseline.run(policy)
        total += reward
        print_stats(baseline.sim)
    print("### Average Rewards", total/n)