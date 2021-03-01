import itertools

import numpy as np
from scipy.stats import truncnorm
import simpy
from gym.spaces import Box
import random


CAR_INTERVAL =  [80 for _ in range(7)] # Avg arrival on the first x hours
CAR_INTERVAL += [10 for _ in range(3)] # Avg arrival on the next ... hours
CAR_INTERVAL += [50 for _ in range(3)] # Avg arrival on the next ... hours
CAR_INTERVAL += [20 for _ in range(2)] # Avg arrival on the next ... hours
CAR_INTERVAL += [40 for _ in range(3)] # Avg arrival on the next ... hours
CAR_INTERVAL += [10 for _ in range(3)] # Avg arrival on the next ... hours
CAR_INTERVAL += [60 for _ in range(3)] # Avg arrival on the next ... hours


GAS_STATION_SIZE = 200  # liters
PUMP_NUMBER = 2  # Number of Pumps


# Mandatory
BASE_CONFIG = {
    "SIM_MINUTES": 5 * 24 * 60, # Simulation time in mINUTES
    "ACTION_INTERVAL_MINUTES": 10, # Time in minutes between each action
    "CAR_TANK_SIZE" : 50 , # liters
    "CAR_TANK_LEVEL" : 8 , # Average tank level, with a exponential distribution
    "REFUELING_SPEED" : 1 , # liters / minute
    "TANK_TRUCK_TIME" : 60 , # Minutes it takes the tank truck to arrive
    "MARGIN_PER_LITRE" : 1 , # Gas margin per litre, excluding truck transportation fixed cost
    "TRUCK_COST" : 150 , # Fixed Transportation Cost
    "DRIVER_PATIENCE" : 4 , # Avg driver patience in minutes, with a exponential distribution
    "CAR_INTERVAL" : CAR_INTERVAL ,
}

class BaseSim(simpy.Environment):
    def __init__(self, config:dict=None):
        super().__init__()
        self.sim_config = BASE_CONFIG.copy()
        if config is not None:
            self.sim_config.update(config)
        self.time = self.now
        self.sim_time = config["SIM_MINUTES"]
        self.step_time = config["ACTION_INTERVAL_MINUTES"]  # Time Units to run between gym env steps

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


N_ACTIONS = 2  # 0 - DoNothing; 1 - Send the Truck
OBSERVATION_SPACE = Box(low=np.array([0, 0, 0] + [0] * 24),
                        high=np.array([GAS_STATION_SIZE, PUMP_NUMBER, 1] + [1] * 24),
                        dtype=np.float64)


class SimModel(BaseSim):
    def __init__(self, config:dict=None):
        super().__init__(config)
        self.fuel_pump = simpy.Container(self, GAS_STATION_SIZE, init=GAS_STATION_SIZE / 2)
        self.gas_station = simpy.Resource(self, PUMP_NUMBER)
        self.process(self.car_generator(self.gas_station, self.fuel_pump))
        self.actual_revenue = 0
        self.last_revenue = 0
        self.free_truck = 1

    def get_observation(self):
        self.run_until_action()
        hour_mask = hot_encode((self.now // 60) % 24, 24)
        env_status = [self.fuel_pump.level, self.gas_station.count, self.free_truck] + hour_mask
        return np.array(env_status)

    def get_reward(self):
        revenue = self.actual_revenue - self.last_revenue
        self.last_revenue = self.actual_revenue
        return revenue, self.done(), {}  # Reward, Done, Info

    # Executes an action
    def exec_action(self, action):
        if action:
            self.process(self.tank_truck(self.fuel_pump))

    # Process: Gas Tank Refuel by a Tank Truck
    def tank_truck(self, fuel_pump):
        """Arrives at the gas station after a certain delay and refuels it."""
        self.free_truck = 0
        yield self.timeout(BASE_CONFIG["TANK_TRUCK_TIME"])
        dprint('Tank truck arriving at time %d' % self.now)
        ammount = fuel_pump.capacity - fuel_pump.level
        dprint('Tank truck refuelling %.1f liters.' % ammount)
        self.actual_revenue -= BASE_CONFIG["TRUCK_COST"]
        self.free_truck = 1
        if ammount > 0:
            yield fuel_pump.put(ammount)

    # Process: Car Refuel
    def car(self, name, gas_station, fuel_pump):
        """A car arrives at the gas station for refueling.
        It requests one of the gas station's fuel pumps and tries to get the
        desired amount of gas from it. If the stations reservoir is
        depleted, the car has to wait for the tank truck to arrive.
        """
        avg, minim, maxim, scale = BASE_CONFIG["CAR_TANK_SIZE"], 0, BASE_CONFIG["CAR_TANK_SIZE"], 10
        fuel_tank_level = truncnorm.rvs((minim-avg)/scale, (maxim-avg)/scale,avg,scale)
        #fuel_tank_level = random.triangular(0,BASE_CONFIG["CAR_TANK_LEVEL"],BASE_CONFIG["CAR_TANK_SIZE"])
        dprint('%s arriving at gas station at %.1f' % (name, self.now))
        with gas_station.request() as req:
            start = self.now
            # Request one of the gas pumps or leave if it take to long
            result = yield req | self.timeout(random.expovariate(1/BASE_CONFIG["DRIVER_PATIENCE"]))

            if req in result:

                # Get the required amount of fuel
                liters_required = BASE_CONFIG["CAR_TANK_SIZE"] - fuel_tank_level
                yield fuel_pump.get(liters_required)

                # Pay the fuel
                self.actual_revenue += liters_required * BASE_CONFIG["MARGIN_PER_LITRE"]

                # The "actual" refueling process takes some time
                yield self.timeout(liters_required / BASE_CONFIG["REFUELING_SPEED"])

                dprint('%s finished refueling in %.1f minutes.' % (name, self.now - start))
            else:
                dprint("{} waited {} minutes and left without refueling".format(name, self.now - start))

    # Generator: Generate car arrivals at the gas station
    def car_generator(self, gas_station: simpy.Resource, fuel_pump: simpy.Container):
        """Generate new cars that arrive at the gas station."""
        for i in itertools.count():
            hour = (self.now // 60) % 24
            yield self.timeout(int(random.expovariate(1/BASE_CONFIG["CAR_INTERVAL"][hour])))
            self.process(self.car('Car %d' % i, gas_station, fuel_pump))

# Mandatory
class SimBaseline:
    def __init__(self, baseline_config:dict=None, sim_config:dict=None):
        self.sim = None
        self.baseline_config = {"level":25}
        if baseline_config is not None:
            self.baseline_config.update(baseline_config)
        self.sim_config = BASE_CONFIG.copy()
        if sim_config is not None:
            self.sim_config.update(sim_config)

    def get_action(self, obs):
        return obs[0] < self.baseline_config["level"] and obs[2]

    def run(self):
        self.sim = SimModel(self.sim_config)
        done = False
        total_reward = 0
        while not done:
            obs = self.sim.get_observation()
            action = self.get_action(obs)
            self.sim.exec_action(action)
            reward, done, _ = self.sim.get_reward()
            total_reward += reward

        return total_reward

def print_stats(sim: SimModel):
    pass

if __name__ == "__main__":
    n = 5
    total = 0
    for _ in range(n):
        baseline = SimBaseline(baseline_config={"level":25})
        reward = baseline.run()
        total += reward
        print_stats(baseline.sim)
    print("### Average Rewards", total/n)
