import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import ray
import ray.rllib.agents.ppo as ppo

from simpy_env import SimpyEnv
from utils import db_connect, TRAINER_DB_NAME, P_MARKER, select_record, SQLParamList, select_all

# from starlette.requests import Request
# from ray import serve
# from ray.serve.exceptions import RayServeException


def cast_non_json(x):
    if isinstance(x, np.float32):
        return float(x)
    elif isinstance(x, dict):
        return {key: cast_non_json(value) for key, value in x.items()}
    return x


def filter_dict(dic_in: dict, keys: set):
    return {key: cast_non_json(dic_in[key]) for key in keys}


def my_ray_init():
    stderrout = sys.stderr
    sys.stderr = open('ray.log', 'w')
    try:
        ray.init(include_dashboard=False, log_to_driver=False, logging_level=0, address='auto')
    except ValueError:
        ray.init(include_dashboard=False, log_to_driver=False, logging_level=0)
    except Exception as e:
        raise e
    sys.stderr = stderrout


def my_ray_train(trainer):
    result = trainer.train()
    return result


class AISimAgent:
    ppo_config = {
        "vf_clip_param"      : 10,  # tune.grid_search([20.0, 100.0]),
        "num_workers"        : 3,
        "num_cpus_per_worker": 0,
        # "lr"            : tune.grid_search([1e-4, 1e-6]),
        "batch_mode"         : "complete_episodes",
        "framework"          : "torch",
        "log_level"          : "ERROR",
    }

    default_sim_config_name = "Base Config"
    default_sim_checkpoint_path = "./checkpoints"

    def __init__(self, sim_name: str, log_level: str = "ERROR", checkpoint_path=None):
        exec_locals = {}
        try:
            exec("from models.{} import SimBaseline, N_ACTIONS, OBSERVATION_SPACE, SimModel, BASE_CONFIG".format(
                sim_name), {},
                exec_locals)
        except ModuleNotFoundError:
            raise Exception(" Model '{}' not found!!".format(sim_name))
        except Exception as e:
            raise e

        try:
            self.db = db_connect(TRAINER_DB_NAME)
        except Exception as e:
            raise e

        assert isinstance(exec_locals['BASE_CONFIG'], dict), "Simulation Config {} must be a dict!".format(
            exec_locals['BASE_CONFIG'])

        assert log_level in ["DEBUG", "INFO", "WARN", "ERROR"], "Invalid log_level {}".format(log_level)

        if not ray.is_initialized():
            my_ray_init()

        self._sim_baseline = exec_locals['SimBaseline']

        sql = '''SELECT id FROM sim_model WHERE name = {}'''.format(P_MARKER)
        params = (sim_name,)
        row = select_record(self.db, sql=sql, params=params)
        if row is None:
            cursor = self.db.cursor()
            cursor.execute('''INSERT INTO sim_model (name) VALUES ({})'''.format(P_MARKER), params)
            self._model_id = cursor.lastrowid
            params = (self._model_id, self.default_sim_config_name,
                      self._get_baseline_avg(exec_locals['BASE_CONFIG']), json.dumps(exec_locals['BASE_CONFIG']))
            cursor.execute('''INSERT INTO sim_config (sim_model_id,
                                                      name,
                                                      baseline_avg,
                                                      config) VALUES ({})'''.format(SQLParamList(4)), params)
            self.db.commit()
            print("# {} Created!".format(sim_name))
        else:
            self._model_id, = row

        self._config = self.ppo_config.copy()
        self._config["log_level"] = log_level
        self._config["env"] = SimpyEnv
        # ToDo: Change the Observation Space to a fucntion that receive a Sim Config as a parameter.
        #  In this part of the code it received exec_locals['BASE_CONFIG']
        self._config["env_config"] = {"n_actions"        : exec_locals['N_ACTIONS'],
                                      "observation_space": exec_locals['OBSERVATION_SPACE'],
                                      "sim_model"        : exec_locals['SimModel'],
                                      "sim_config"       : exec_locals['BASE_CONFIG']}
        if checkpoint_path is None:
            self.checkpoint_path = self.default_sim_checkpoint_path

    def __del__(self):
        ray.shutdown()

    def _add_session(self, session_data: tuple):
        agent_config, sim_config_id = session_data
        agent_config.pop("env", None)
        agent_config.pop("env_config", None)
        cursor = self.db.cursor()
        _session_data = (self._model_id, sim_config_id, datetime.now(), json.dumps(agent_config))
        cursor.execute('''INSERT INTO training_session (
                                        sim_model_id,
                                        sim_config_id,
                                        time_start,
                                        agent_config) VALUES ({})'''.format(SQLParamList(4)), _session_data)
        self.db.commit()
        return cursor.lastrowid

    def _get_sim_base_config(self):
        sql = '''SELECT id FROM sim_config 
                 WHERE sim_model_id = {} and name = {}'''.format(P_MARKER, P_MARKER)
        params = (self._model_id, self.default_sim_config_name)
        row = select_record(self.db, sql=sql, params=params)
        assert row is not None, "Base Sim Config not found!"
        return row[0]

    def _get_sim_config(self, sim_config: dict):
        cursor = self.db.cursor()
        if sim_config is None:
            sim_config_id = self._get_sim_base_config()
        else:
            sql = '''SELECT id, config FROM sim_config 
                     WHERE sim_model_id = {}'''.format(P_MARKER)
            params = (self._model_id,)
            row_list = select_all(self.db, sql=sql, params=params)
            try:
                idx = [json.loads(config) for _, config in row_list].index(sim_config)
                sim_config_id, _ = row_list[idx]
            except Exception:
                params = (self._model_id, "Config {}".format(len(row_list)),
                          self._get_baseline_avg(sim_config), json.dumps(sim_config))
                cursor.execute('''INSERT INTO sim_config (sim_model_id,
                                                          name,
                                                          baseline_avg,
                                                          config) VALUES ({})'''.format(SQLParamList(4)), params)
                sim_config_id = cursor.lastrowid
        self.db.commit()
        return sim_config_id

    def _get_baseline_avg(self, sim_config: dict):
        @ray.remote
        def base_run(baseline):
            return baseline.run()

        base = self._sim_baseline(sim_config=sim_config)
        return np.mean(ray.get([base_run.remote(base) for _ in range(30)]))

    def _update_session(self, best_policy, duration, session_id):
        cursor = self.db.cursor()
        cursor.execute('''UPDATE training_session SET best_policy = {}, duration = {}
                          WHERE id = {}'''.format(P_MARKER, P_MARKER, P_MARKER), (best_policy, duration, session_id))
        self.db.commit()

    def _add_iteration(self, n, session_id, start_time, best_checkpoint, result):
        cursor = self.db.cursor()
        iteration_other_data_keys = {'info', 'training_iteration', 'experiment_id', 'date', 'timestamp',
                                     'time_this_iter_s'}
        iteration_data = (session_id, n, result['episode_reward_mean'], result['episode_reward_min'],
                          result['episode_reward_max'], best_checkpoint, (datetime.now() - start_time).total_seconds(),
                          start_time, json.dumps(filter_dict(result, iteration_other_data_keys)))
        cursor.execute('''INSERT INTO training_iteration (
                                        training_session_id,
                                        id,
                                        reward_mean,
                                        reward_min,
                                        reward_max,
                                        checkpoint,
                                        duration,
                                        time_start,
                                        other_data) VALUES ({})'''.format(SQLParamList(9)), iteration_data)
        self.db.commit()
        return cursor.lastrowid

    def _add_policy(self, policy_data: tuple):
        cursor = self.db.cursor()
        session_id, best_iteration, best_checkpoint, agent_config, sim_config_id = policy_data
        agent_config.pop("env", None)
        agent_config.pop("env_config", None)
        agent_config = json.dumps(agent_config)
        policy_data = (self._model_id, sim_config_id, session_id, best_iteration, best_checkpoint, agent_config)
        cursor.execute('''INSERT INTO policy (
                                        sim_model_id,
                                        sim_config_id,
                                        session_id,
                                        iteration_id,
                                        checkpoint,
                                        agent_config) VALUES ({})'''.format(SQLParamList(6)), policy_data)
        self.db.commit()
        return cursor.lastrowid

    def _add_policy_run(self, policy_run_data: tuple):
        cursor = self.db.cursor()
        cursor.execute('''INSERT INTO policy_run (
                                        policy_id,
                                        time_start,
                                        simulations,
                                        duration,
                                        results) VALUES ({})'''.format(SQLParamList(5)), policy_run_data)
        self.db.commit()
        return cursor.lastrowid

    def _add_baseline_run(self, policy_run_data: tuple):
        cursor = self.db.cursor()
        cursor.execute('''INSERT INTO baseline_run (
                                        sim_config_id,
                                        time_start,
                                        simulations,
                                        duration,
                                        results) VALUES ({})'''.format(SQLParamList(5)), policy_run_data)
        self.db.commit()
        return cursor.lastrowid

    # ToDo: Retrain
    # ToDo: Add more than one best policy
    # ToDo: Add labels to the sessions
    def train(self, iterations: int = 10, ai_config: dict = None, sim_config: dict = None,
              add_best_policy: bool = True, checkpoint_path=None):

        _agent_config = self._config.copy()

        if ai_config is not None:
            assert isinstance(ai_config, dict), "Agent Config {} must be a dict!".format(ai_config)
            ai_config.pop("env", None)
            ai_config.pop("env_config", None)
            _agent_config.update(ai_config)

        if sim_config is not None:
            assert isinstance(sim_config, dict), "Sim Config {} must be a dict!".format(sim_config)
            # ToDo: Change the Observation Space to a function that receive a Sim Config as a parameter.
            #  In this part of the code the _agent_config["env_config"]["observation_space"] have to be updated
            _agent_config["env_config"]["sim_config"].update(sim_config)

        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        sim_config_id = self._get_sim_config(_agent_config["env_config"]["sim_config"])

        session_id = self._add_session((_agent_config.copy(), sim_config_id))

        print("# Training Session {} started at {}!".format(session_id, datetime.now()))

        trainer = ppo.PPOTrainer(config=_agent_config)

        # print("# DEBUG: Trainer Created at {}!".format(datetime.now()))

        session_start = datetime.now()
        iteration_start = datetime.now()

        result = my_ray_train(trainer)
        # print("# DEBUG: Trainer Result {} at {}!".format(result['episode_reward_mean'], datetime.now()))
        best_checkpoint = trainer.save(checkpoint_dir=checkpoint_path)
        best_reward = result['episode_reward_mean']
        print("# Progress: {:2.1%} # Best Mean Reward: {:.2f}      ".format(1 / iterations, best_reward), end="\r")
        self._add_iteration(0, session_id, iteration_start, best_checkpoint, result)
        best_iteration = 0

        for i in range(1, iterations):
            iteration_start = datetime.now()
            result = my_ray_train(trainer)
            # result = trainer.train()

            if result['episode_reward_mean'] > best_reward:
                best_checkpoint = trainer.save(checkpoint_dir=checkpoint_path)
                best_reward = result['episode_reward_mean']
                best_iteration = i
                checkpoint = best_checkpoint
            else:
                checkpoint = None
            print("# Progress: {:2.1%} # Best Mean Reward: {:.2f}      ".format((i + 1) / iterations, best_reward),
                  end="\r")
            self._add_iteration(i, session_id, iteration_start, checkpoint, result)

        print("# Progress: {:2.1%} # Best Mean Reward: {:.2f}      ".format(1, best_reward))
        self._update_session(best_iteration, (datetime.now() - session_start).total_seconds(), session_id)

        if add_best_policy:
            policy_data = (session_id, best_iteration, best_checkpoint,
                           _agent_config.copy(), sim_config_id)
            self._add_policy(policy_data)

        # ray.shutdown()

        print("# Training Session {} ended at {}!".format(session_id, datetime.now()))

    def del_training_sessions(self, sessions: [int, list] = None):
        select_sessions_sql = '''SELECT id FROM training_session
                                 WHERE sim_model_id = {}'''.format(P_MARKER)
        params = (self._model_id,)
        all_sessions = select_all(self.db, sql=select_sessions_sql, params=params)
        all_sessions = {t[0] for t in all_sessions}
        del_sessions = []
        if isinstance(sessions, int):
            assert sessions in all_sessions, "Invalid session id {}".format(sessions)
            del_sessions = (sessions,)
        if isinstance(sessions, list):
            assert set(sessions).issubset(all_sessions), "Invalid sessions list {}".format(sessions)
            del_sessions = tuple(sessions)
        if sessions is None:
            del_sessions = tuple(all_sessions)
        if len(del_sessions):
            cursor = self.db.cursor()
            sql = '''DELETE FROM training_iteration
                     WHERE training_session_id IN ({})'''.format(SQLParamList(len(del_sessions)))
            cursor.execute(sql, del_sessions)
            sql = '''DELETE FROM training_session
                     WHERE id IN ({})'''.format(SQLParamList(len(del_sessions)))
            cursor.execute(sql, del_sessions)
            self.db.commit()

    '''
    Training Performance
    - A Line Graph showing the mean_rewards of all (or some) of the Training Sessions of a specific Model 
    '''

    def get_training_sessions(self):
        sql = '''SELECT training_session_id 
                 FROM training_iteration
                 WHERE sim_model_id = {}'''.format(P_MARKER)
        params = (self._model_id,)
        df = pd.read_sql_query(sql, self.db, params=params)
        return df

    def get_sim_config(self):
        sql = '''SELECT id, name, baseline_avg, config
                 FROM sim_config
                 WHERE sim_model_id = {}'''.format(P_MARKER)
        params = (self._model_id,)
        df = pd.read_sql_query(sql, self.db, params=params)
        return df

    # ToDo: Change to allow more than one config
    def get_training_data(self, sim_config: int = None, baseline: bool = True):

        if sim_config is None:
            sim_config = self._get_sim_base_config()
        else:
            sql = "SELECT id FROM sim_config WHERE id = {}".format(P_MARKER)
            row = select_record(self.db, sql=sql, params=(sim_config,))
            assert row is not None, "Invalid Sim Config id {}".format(sim_config)
            sim_config, = row

        sql = '''SELECT training_session_id as session, training_iteration.id as iteration, reward_mean 
                 FROM training_iteration
                 INNER JOIN training_session ON training_iteration.training_session_id = training_session.id
                 INNER JOIN sim_config ON training_session.sim_config_id = sim_config.id
                 WHERE training_session.sim_config_id = {}'''.format(P_MARKER)
        params = (sim_config,)
        df = pd.read_sql_query(sql, self.db, params=params) \
            .pivot(index='iteration', columns='session', values='reward_mean')

        if baseline:
            sql = "SELECT baseline_avg FROM sim_config WHERE id = {}".format(P_MARKER)
            baseline_avg, = select_record(self.db, sql=sql, params=(sim_config,))
            df['baseline'] = [baseline_avg for _ in range(df.shape[0])]

        return df

    def get_policies(self):
        sql = '''SELECT sim_config_id as sim_config, id as policy, session_id as session
                 FROM policy
                 WHERE sim_model_id = {}'''.format(P_MARKER)
        return pd.read_sql_query(sql, self.db, params=(self._model_id,))

    def run_policies(self, policy: [int, list] = None, simulations: int = 1):

        select_policy_sql = '''SELECT id FROM policy
                                 WHERE sim_model_id = {}'''.format(P_MARKER)
        all_policies = select_all(self.db, sql=select_policy_sql, params=(self._model_id,))
        all_policies = {t[0] for t in all_policies}

        if isinstance(policy, int):
            assert policy in all_policies, "Invalid session id {}".format(policy)
            policies = (policy,)
        elif isinstance(policy, list):
            assert set(policy).issubset(all_policies), "Invalid sessions list {}".format(policy)
            policies = tuple(policy)
        else:
            policies = tuple(all_policies)

        select_policy_sql = '''SELECT policy.id, checkpoint, agent_config, sim_config.config as s_config
                               FROM policy INNER JOIN sim_config ON policy.sim_config_id = sim_config.id
                               WHERE policy.id IN ({})'''.format(SQLParamList(len(policies)))
        policy_data = select_all(self.db, sql=select_policy_sql, params=policies)

        for policy_id, checkpoint, saved_agent_config, saved_sim_config in policy_data:

            print("# Running AI Policy {} started at {}!".format(policy_id, datetime.now()))

            agent_config = self._config.copy()
            agent_config.update(json.loads(saved_agent_config))

            sim_config = json.loads(saved_sim_config)

            agent = ppo.PPOTrainer(config=agent_config)
            agent.restore(checkpoint)

            time_start = datetime.now()
            # instantiate env class
            agent_config["env_config"]["sim_config"].update(sim_config)
            he = SimpyEnv(agent_config["env_config"])

            result_list = []
            for i in range(simulations):
                # run until episode ends
                episode_reward = 0
                done = False
                obs = he.reset()
                while not done:
                    action = agent.compute_action(obs)
                    obs, reward, done, info = he.step(action)
                    episode_reward += reward
                result_list.append(episode_reward)
                print("# Progress: {:2.1%} ".format((i + 1) / simulations), end="\r")
            policy_run_data = (policy_id, time_start, simulations,
                               (datetime.now() - time_start).total_seconds(), json.dumps(result_list))
            self._add_policy_run(policy_run_data)
            print("# Progress: {:2.1%} ".format(1))
            print("# Running AI Policy {} ended at {}!".format(policy_id, datetime.now()))

        # ray.shutdown()

    # Todo: Add Baselines config
    def run_baselines(self, sim_config: [int, list] = None, simulations: int = 1):

        @ray.remote
        def base_run(base):
            return base.run()

        if sim_config is None:
            # Get all sim configs for the current model
            select_sim_sql = '''SELECT id, config FROM sim_config
                                WHERE sim_config.sim_model_id = {}'''.format(P_MARKER)
            rows = select_all(self.db, sql=select_sim_sql, params=(self._model_id,))
            sim_configs = ((i, json.loads(config)) for i, config in rows)
        else:
            if isinstance(sim_config, int):
                sim_config = [sim_config]
            if isinstance(sim_config, list):
                # Get all policies for the list of sim_configs
                select_sim_sql = '''SELECT id, config FROM sim_config
                                    WHERE id IN ({})'''.format(SQLParamList(len(sim_config)))
                rows = select_all(self.db, sql=select_sim_sql, params=tuple(sim_config))
                sim_configs = ((i, json.loads(config)) for i, config in rows)
            else:
                raise Exception("Invalid Sim Config {}".format(sim_config))

        for sim_config_id, sim_config in sim_configs:
            base = self._sim_baseline(sim_config=sim_config)
            print("# Baseline Simulation for Config {} started at {}!".format(sim_config_id, datetime.now()))
            time_start = datetime.now()
            result_list = ray.get([base_run.remote(base) for _ in range(simulations)])
            # for i in range(simulations):
            #    future_result_list.append(base_run.remote())
            #    print("# Progress: {:2.1%} ".format((i + 1) / simulations), end="\r")

            policy_run_data = (sim_config_id, time_start, simulations,
                               (datetime.now() - time_start).total_seconds(), json.dumps(result_list))
            self._add_baseline_run(policy_run_data)
            # print("# Progress: {:2.1%} ".format(1))
            print("# Baseline Simulation for Config {} ended at {}!".format(sim_config_id, datetime.now()))

    def get_policy_run_data(self, sim_config: int = None, baseline: bool = True):

        if sim_config is None:
            sim_config = self._get_sim_base_config()
        else:
            sql = "SELECT id FROM sim_config WHERE id = {}".format(P_MARKER)
            row = select_record(self.db, sql=sql, params=(sim_config,))
            assert row is not None, "Invalid Sim Config id {}".format(sim_config)
            sim_config, = row

        sql = '''SELECT policy_id, policy_run.id, time_start, results 
                 FROM policy_run
                 INNER JOIN policy ON policy_run.policy_id = policy.id
                 WHERE policy.sim_config_id = {}'''.format(P_MARKER)
        params = (sim_config,)
        policy_run = select_all(self.db, sql=sql, params=params)
        df = pd.DataFrame([["ai_policy{}_run{}".format(policy_id, run_id), time, x]
                           for policy_id, run_id, time, l in policy_run for x in json.loads(l)],
                          columns=['policy', 'time', 'reward'])
        if baseline:
            sql = '''SELECT id, time_start, results 
                     FROM baseline_run
                     WHERE sim_config_id = {}'''.format(P_MARKER)
            params = (sim_config,)
            baseline_run = select_all(self.db, sql=sql, params=params)
            df2 = pd.DataFrame([["baseline_run{}".format(run_id), time, x]
                                for run_id, time, l in baseline_run for x in json.loads(l)],
                               columns=['policy', 'time', 'reward'])
            df = df.append(df2)

        return df



