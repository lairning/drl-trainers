import ray
import ray.rllib.agents.ppo as ppo
from datetime import datetime
import statistics as stats
import json
import numpy as np

from simpy_env import SimpyEnv

from utils import db_connect, DB_NAME, P_MARKER, select_record, SQLParamList

def cast_non_json(x):
    if isinstance(x,np.float32):
        return float(x)
    elif isinstance(x,dict):
        return {key:cast_non_json(value) for key,value in x.items()}
    return x

def filter_dict(dic_in: dict, keys: set):
    return {key:cast_non_json(dic_in[key]) for key in keys}


class AISimAgent():
    ppo_config = {
        "vf_clip_param": 10,  # tune.grid_search([20.0, 100.0]),
        "num_workers"  : 5,
        # "lr"            : tune.grid_search([1e-4, 1e-6]),
        "batch_mode"   : "complete_episodes",
        "framework"    : "torch",
        "log_level"    : "ERROR"
    }

    def __init__(self, sim_name: str, agent_config=None):
        exec_locals = {}
        try:
            exec("from models.{} import SimBaseline, N_ACTIONS, OBSERVATION_SPACE, SimModel".format(sim_name),{},exec_locals)
        except Exception as e:
            raise e

        try:
            self.db = db_connect(DB_NAME)
        except Exception as e:
            raise e

        sql = '''SELECT id FROM sim_model WHERE name = {}'''.format(P_MARKER)
        params = (sim_name,)
        row = select_record(self.db, sql=sql, params=params)
        if row is None:
            cursor = self.db.cursor()
            cursor.execute('''INSERT INTO sim_model (name) VALUES ({})'''.format(P_MARKER),params)
            print("# {} Created!".format(sim_name))
            self._model_id = cursor.lastrowid
            self.db.commit()
        else:
            self._model_id, = row

        if agent_config is None:
            agent_config = {}
        else:
            assert isinstance(agent_config, dict), "Config {} must be a dict!".format(agent_config)
        self._config = self.ppo_config.copy()
        self._config.update(agent_config)
        self._config["env"] = SimpyEnv
        self._config["env_config"] = {"n_actions" : exec_locals['N_ACTIONS'],
                                 "observation_space" : exec_locals['OBSERVATION_SPACE'],
                                 "sim_model" : exec_locals['SimModel']}

    def _add_session(self, session_data: tuple):
        cursor = self.db.cursor()
        config = session_data[2].copy()
        config.pop("env", None)
        config.pop("env_config", None)
        _session_data = (session_data[0], session_data[1], json.dumps(config))
        cursor.execute('''INSERT INTO training_session (
                                        sim_model_id,
                                        time_start,
                                        config) VALUES ({})'''.format(SQLParamList(3)), _session_data)
        self.db.commit()
        return cursor.lastrowid

    def _update_session(self, best_policy, duration):
        cursor = self.db.cursor()
        cursor.execute('''UPDATE training_session SET best_policy = {}, duration = {}
                          WHERE id = {}'''.format(P_MARKER,P_MARKER,P_MARKER), (best_policy, duration,
                                                                                self._training_session_id))
        self.db.commit()

    def _add_iteration(self, session_id, start_time, best_checkpoint, result):
        cursor = self.db.cursor()
        iteration_other_data_keys = {'info', 'training_iteration','experiment_id', 'date', 'timestamp', 'time_this_iter_s'}
        iteration_data = (session_id, result['episode_reward_mean'], result['episode_reward_min'],
                          result['episode_reward_mean'], best_checkpoint, (datetime.now()-start_time).total_seconds(),
                          start_time, json.dumps(filter_dict(result,iteration_other_data_keys)))
        print(iteration_data)
        cursor.execute('''INSERT INTO training_iteration (
                                        training_session_id,
                                        reward_mean,
                                        reward_min,
                                        reward_max,
                                        checkpoint,
                                        duration,
                                        time_start,
                                        other_data) VALUES ({})'''.format(SQLParamList(8)), iteration_data)
        self.db.commit()
        return cursor.lastrowid


    def train(self, sessions: int = 10, config=None):

        if config is None:
            config = {}
        else:
            assert isinstance(config, dict), "Config {} must be a dict!".format(config)
            config.pop("env",None)
            config.pop("env_config",None)

        _config = self._config.copy()
        _config.update(config)

        session_start = datetime.now()
        session_data = (self._model_id, session_start, _config)
        self._training_session_id = self._add_session(session_data)

        print("# Training Session {} started at {}!".format(self._training_session_id, datetime.now()))


        ray.init()

        self._trainer = ppo.PPOTrainer(config=_config)

        iteration_start = datetime.now()

        result = self._trainer.train()
        best_checkpoint = self._trainer.save()
        best_reward = result['episode_reward_mean']
        print("# Progress: {:2.1%} # Best Mean Reward: {:.2f}      ".format(1/sessions,best_reward), end="\r")
        best_iteration = 0
        best_policy = self._add_iteration(self._training_session_id, iteration_start, best_checkpoint, result)

        for i in range(1, sessions):
            iteration_start = datetime.now()
            result = self._trainer.train()

            if result['episode_reward_mean'] > best_reward:
                best_iteration = i
                best_checkpoint = self._trainer.save()
                best_reward = result['episode_reward_mean']
            else:
                best_checkpoint = None
            print("# Progress: {:2.1%} # Best Mean Reward: {:.2f}      ".format((i+1) / sessions, best_reward),
                  end="\r")
            best_policy = self._add_iteration(self._training_session_id, iteration_start, best_checkpoint, result)


        self._update_session(best_policy, datetime.now()-session_start)

        ray.shutdown()

        print("# Training Session {} ended at {}!".format(self._training_session_id, datetime.now()))

        return best_policy

    def run(self, simulations: int = 1, training_session=None, baseline=None, baseline_policy=None):

        if training_session is None:
            training_session = len(self.training_session)-1

        assert len(self.training_session) >= training_session, \
            "'training_session' {} is higher than the current number of {} Training Sessions ".\
                format(training_session,len(self.training_session))


        config = self.training_session[training_session]['config']
        best_iteration = self.training_session[training_session]['best_iteration']
        check_point = self.training_session[training_session]['result'][best_iteration]['check_point']

        ray.init()

        agent = ppo.PPOTrainer(config=config)
        agent.restore(check_point)

        # instantiate env class
        he = SimpyEnv(config["env_config"])

        result_list = []
        for _ in range(simulations):
            # run until episode ends
            episode_reward = 0
            done = False
            obs = he.reset()
            while not done:
                action = agent.compute_action(obs)
                obs, reward, done, info = he.step(action)
                episode_reward += reward
            result_list.append(episode_reward)
        self.running_session.append({"type": "AI", "result": result_list})

        if baseline is not None:
            result_list = [baseline.run(baseline_policy) for _ in range(simulations)]
            self.running_session.append({"type": "Baseline", "result": result_list})

        ray.shutdown()

        return self.running_session


