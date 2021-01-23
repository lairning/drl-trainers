import itertools

import gym
import numpy as np
import simpy
from gym.spaces import Discrete, Box, Dict


class BaseSim(simpy.Environment):
    def __init__(self, step_time=1):
        super().__init__()

        self.time = self.now
        self.step_time = step_time  # Time Units to run between gym env steps

    def run_until_action(self):
        self.run(until=self.time + self.step_time)
        self.time = self.now

    def exec_action(self, action):
        raise Exception("Not Implemented!!!")

    def get_observation(self):
        raise Exception("Not Implemented!!!")

    def get_reward(self):
        raise Exception("Not Implemented!!!")


GAS_STATION_SIZE = 200  # liters
CAR_TANK_SIZE = 50  # liters
CAR_TANK_LEVEL = [2, 20]  # Min/max levels of fuel tanks (in liters)
REFUELING_SPEED = 1  # liters / minute
TANK_TRUCK_TIME = 60  # Minutes it takes the tank truck to arrive
PUMP_NUMBER = 2  # Number of Pumps
MARGIN_PER_LITRE = 1  # Gas margin er litre, excluding truck transportation fixed cost
TRUCK_COST = 150  # Fixed Transportation Cost
DRIVER_PATIENCE = [1, 8]  # Min/max level of driver patience
CAR_INTERVAL = [[40, 120] for _ in range(7)] + [[5, 10] for _ in range(3)] + [[20, 80] for _ in range(3)]
CAR_INTERVAL += [[5, 40] for _ in range(2)] + [[20, 80] for _ in range(3)]
CAR_INTERVAL += [[5, 20] for _ in range(3)] + [[20, 80] for _ in range(3)]

DEBUG = False


def dprint(*args):
    if DEBUG:
        print(*args)

def hot_encode(n, N):
    encode = [0]*N
    encode[n] = 1
    return encode

class Sim(BaseSim):
    def __init__(self, sim_time, step_time=1):
        super().__init__(step_time)
        self.sim_time = sim_time
        self.fuel_pump = simpy.Container(self, GAS_STATION_SIZE, init=GAS_STATION_SIZE / 2)
        self.gas_station = simpy.Resource(self, PUMP_NUMBER)
        self.process(self.car_generator(self.gas_station, self.fuel_pump))
        self.actual_revenue = 0
        self.last_revenue = 0
        # self.truck_revenue = 0
        self.free_truck = 1

    def get_observation(self):
        hour_mask = hot_encode((self.now // 60) % 24, 24)
        env_status = [self.fuel_pump.level, self.gas_station.count, self.free_truck]+hour_mask
        return np.array(env_status)

    def get_reward(self):
        revenue = self.actual_revenue - self.last_revenue
        self.last_revenue = self.actual_revenue
        # revenue = self.truck_revenue
        # self.truck_revenue = 0
        done = self.now >= self.sim_time
        return revenue, done, {}  # Reward, Done, Info

    def exec_action(self, action):
        if action:
            self.process(self.tank_truck(self.fuel_pump))

    def tank_truck(self, fuel_pump):
        """Arrives at the gas station after a certain delay and refuels it."""
        self.free_truck = 0
        yield self.timeout(TANK_TRUCK_TIME)
        dprint('Tank truck arriving at time %d' % self.now)
        ammount = fuel_pump.capacity - fuel_pump.level
        dprint('Tank truck refuelling %.1f liters.' % ammount)
        self.actual_revenue -= TRUCK_COST
        # env.truck_revenue = env.actual_revenue - env.last_revenue
        # env.last_revenue = env.actual_revenue
        self.free_truck = 1
        if ammount > 0:
            yield fuel_pump.put(ammount)


    def car(self, name, gas_station, fuel_pump):
        """A car arrives at the gas station for refueling.
        It requests one of the gas station's fuel pumps and tries to get the
        desired amount of gas from it. If the stations reservoir is
        depleted, the car has to wait for the tank truck to arrive.
        """
        fuel_tank_level = np.random.randint(*CAR_TANK_LEVEL)
        dprint('%s arriving at gas station at %.1f' % (name, self.now))
        with gas_station.request() as req:
            start = self.now
            # Request one of the gas pumps or leave if it take to long
            result = yield req | self.timeout(np.random.randint(*DRIVER_PATIENCE))

            if req in result:

                # Get the required amount of fuel
                liters_required = CAR_TANK_SIZE - fuel_tank_level
                yield fuel_pump.get(liters_required)

                # Pay the fuel
                self.actual_revenue += liters_required * MARGIN_PER_LITRE

                # The "actual" refueling process takes some time
                yield self.timeout(liters_required / REFUELING_SPEED)

                dprint('%s finished refueling in %.1f minutes.' % (name, self.now - start))
            else:
                dprint("{} waited {} minutes and left without refueling".format(name, self.now - start))


    def car_generator(self, gas_station: simpy.Resource, fuel_pump: simpy.Container):
        """Generate new cars that arrive at the gas station."""
        for i in itertools.count():
            hour = (self.now // 60) % 24
            yield self.timeout(np.random.randint(*CAR_INTERVAL[hour]))
            self.process(self.car('Car %d' % i, gas_station, fuel_pump))

N_ACTIONS = 2  # 0 - DoNothing; 1 - Send the Truck
SIM_TIME = 5 * 24 * 60  # Simulation time in Time units (minutes)
STEP_TIME = 10  # Time units (minutes) between each step

OBSERVATION_SPACE = Box(low=np.array([0, 0, 0]+[0]*24),
                             high=np.array([GAS_STATION_SIZE, PUMP_NUMBER, 1]+[1]*24),
                             dtype=np.float64)

class SimpyEnv(gym.Env):

    def __init__(self):
        self.action_space = Discrete(N_ACTIONS)
        self.observation_space = OBSERVATION_SPACE
        self.sim = None

    def reset(self):
        self.sim = Sim(sim_time=SIM_TIME, step_time=STEP_TIME)

        # Start processes and initialize resources
        self.sim.run_until_action()
        obs = self.sim.get_observation()
        #assert self.observation_space.contains(obs), "{} not in {}".format(obs, self.observation_space)
        return obs

    def step(self, action):
        assert action in range(self.action_space.n)

        self.sim.exec_action(action)
        self.sim.run_until_action()
        obs = self.sim.get_observation()
        reward, done, info = self.sim.get_reward()

        #assert self.observation_space.contains(obs), "{} not in {}".format(obs, self.observation_space)
        return obs, reward, done, info


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

#for level in [45,55,65,75,85]:
for level in [65]:
    N = 100
    total = 0
    for i in range(N):
        # env_ = SimpyEnv()
        env = SimpyEnv()
        obs = env.reset()
        done = False
        while not done:
            dprint(obs)
            action = obs[0] < level and obs[2]
            #action = np.random.randint(N_ACTIONS)
            obs, reward, done, info = env.step(action)
            dprint("Decision {} with {} revenue".format(action, reward))
        total += env.sim.actual_revenue
    print("Average Revenue for level {}:".format(level), total/ N)

