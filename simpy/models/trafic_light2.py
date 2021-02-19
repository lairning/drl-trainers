import itertools

import numpy as np
import simpy
from gym.spaces import Box
import random

# SIM_TIME = 1 * 24 * 60 * 60  # Simulation time in Time units (seconds)
SIM_TIME = 1 * 2 * 60 * 60  # Simulation time in Time units (seconds)
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
MTBC_BASE = [40, 30, 50, 60, 40, 20, 70, 60]
# Mean Time Between Cars
MTBC = [x * 1 for x in MTBC_BASE]

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
    def __init__(self):
        super().__init__()
        self.lights = [Light(LIGHTS[i], self, STATUS[0][i], MTBC[i]) for i in range(len(LIGHTS))]
        self.total_reward = 0

    def get_observation(self):
        self.run_until_action()
        obs = []
        for light in self.lights:
            obs += light.get_observation()
        return obs

    def get_reward(self):
        total_reward = 0
        l_cars_q = [self.now-value for light in self.lights for value in light.queue.values()]
        if len(l_cars_q):
            total_reward = - sum(l_cars_q)*(1+ np.std(l_cars_q) / np.mean(l_cars_q))
        # total_reward = - sum((self.now-value) for light in self.lights for value in light.queue.values() )
        # total_reward += - sum(sum(light.stats['waiting_time']) for light in self.lights)
        reward = total_reward - self.total_reward
        self.total_reward = total_reward
        return reward, self.done(), {}  # Reward, Done, Info

    # Executes an action
    def exec_action(self, action):
        # An action set the status of the lights according to the STATUS Table
        for i, light in enumerate(self.lights):
            self.process(light.set_status(STATUS[action][i]))
        self.current_status_id = action


class SimBaseline:
    def __init__(self):
        self.sim = None

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

    def run(self, policy = RoundRobin(6)):
        self.sim = SimModel()
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
    n = 10
    total = 0
    for _ in range(n):
        baseline = SimBaseline()
        policy = baseline.RoundRobin(1)
        reward = baseline.run(policy)
        total += reward
        print_stats(baseline.sim)
    print("### Average Rewards", total/n)