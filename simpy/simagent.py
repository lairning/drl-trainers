import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import ray
import ray.rllib.agents.ppo as ppo

from simpy_env import SimpyEnv
from utils import db_connect, DB_NAME, P_MARKER, select_record, SQLParamList, select_all


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
    ray.init(include_dashboard=False, log_to_driver=False, logging_level=0)
    sys.stderr = stderrout


class AISimAgent:
    ppo_config = {
        "vf_clip_param": 10,  # tune.grid_search([20.0, 100.0]),
        "num_workers"  : 5,
        # "lr"            : tune.grid_search([1e-4, 1e-6]),
        "batch_mode"   : "complete_episodes",
        "framework"    : "torch"
    }

    default_sim_config_name = "Base Config"

    def __init__(self, sim_name: str):
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
            self.db = db_connect(DB_NAME)
        except Exception as e:
            raise e

        assert isinstance(exec_locals['BASE_CONFIG'], dict), "Simulation Config {} must be a dict!".format(
            exec_locals['BASE_CONFIG'])

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
        self._config["env"] = SimpyEnv
        self._config["env_config"] = {"n_actions"        : exec_locals['N_ACTIONS'],
                                      "observation_space": exec_locals['OBSERVATION_SPACE'],
                                      "sim_model"        : exec_locals['SimModel'],
                                      "sim_config"       : exec_locals['BASE_CONFIG']}

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
                idx = [json.loads(row[1]) for row in row_list].index(sim_config)
                sim_config_id = [idx][0]
            except Exception:
                params = (self._model_id, "Config {}".format(len(row_list)),
                          self._get_baseline_avg(sim_config), json.dumps(sim_config))
                cursor.execute('''INSERT INTO sim_config (sim_model_id
                                                          name,
                                                          baseline_avg,
                                                          config) VALUES ({})'''.format(SQLParamList(4)), params)
                sim_config_id = cursor.lastrowid
        self.db.commit()
        return sim_config_id

    # ToDo: Explore the use of Ray to speed up this operation
    def _get_baseline_avg(self, sim_config: dict):
        base = self._sim_baseline()
        return np.mean([base.run(sim_config=sim_config) for _ in range(30)])

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

    # ToDo: Add more than one best policy
    # ToDo: Add labels to the sessions
    def train(self, iterations: int = 10, agent_config: dict = None, sim_config: dict = None,
              add_best_policy: bool = True):

        _agent_config = self._config.copy()

        if agent_config is not None:
            assert isinstance(agent_config, dict), "Agent Config {} must be a dict!".format(agent_config)
            agent_config.pop("env", None)
            agent_config.pop("env_config", None)
            _agent_config.update(agent_config)

        if sim_config is not None:
            assert isinstance(sim_config, dict), "Sim Config {} must be a dict!".format(sim_config)
            _agent_config["env_config"]["sim_config"].update(sim_config)

        sim_config_id = self._get_sim_config(_agent_config["env_config"]["sim_config"])

        session_id = self._add_session((_agent_config.copy(), sim_config_id))

        my_ray_init()

        print("# Training Session {} started at {}!".format(session_id, datetime.now()))

        trainer = ppo.PPOTrainer(config=_agent_config)

        session_start = datetime.now()
        iteration_start = datetime.now()

        result = trainer.train()
        best_checkpoint = trainer.save()
        best_reward = result['episode_reward_mean']
        print("# Progress: {:2.1%} # Best Mean Reward: {:.2f}      ".format(1 / iterations, best_reward), end="\r")
        self._add_iteration(0, session_id, iteration_start, best_checkpoint, result)
        best_iteration = 0

        for i in range(1, iterations):
            iteration_start = datetime.now()
            result = trainer.train()

            if result['episode_reward_mean'] > best_reward:
                best_checkpoint = trainer.save()
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

        ray.shutdown()

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

    def get_training_data(self, sim_config: int=None, baseline: bool = True):

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
        sql = '''SELECT id as policy, session_id as session
                 FROM policy
                 WHERE sim_model_id = {}'''.format(P_MARKER)
        return pd.read_sql_query(sql, self.db, params=(self._model_id,))

    def run_ai_policies(self, policy: [int, list] = None, simulations: int = 1):

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

        select_policy_sql = '''SELECT id, checkpoint, agent_config, sim_config.config as s_config
                               FROM policy INNER JOIN sim_config ON policy.sim_config_id = sim_config.id
                               WHERE id IN ({})'''.format(SQLParamList(len(policies)))
        policy_data = select_all(self.db, sql=select_policy_sql, params=policies)

        my_ray_init()

        for policy_id, checkpoint, saved_agent_config, saved_sim_config in policy_data:

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
            policy_run_data = (policy_id, time_start, simulations,
                               (datetime.now() - time_start).total_seconds(), json.dumps(result_list))
            self._add_policy_run(policy_run_data)

        ray.shutdown()

    # ToDo: Implement
    def run_baseline_policies(self, sim_config: [int, list] = None, simulations: int = 1):
        pass

    def get_policy_run_data(self, sim_config: int = None, baseline: bool = False):

        if sim_config is None:
            sim_config = self._get_sim_base_config()
        else:
            sql = "SELECT id FROM sim_config WHERE id = {}".format(P_MARKER)
            row = select_record(self.db, sql=sql, params=(sim_config,))
            assert row is not None, "Invalid Sim Config id {}".format(sim_config)
            sim_config, = row

        sql = '''SELECT policy_id as policy, time_start, results 
                 FROM policy_run
                 INNER JOIN policy ON policy_run.policy_id = policy.id
                 WHERE policy.sim_config_id = {}'''.format(P_MARKER)
        params = (sim_config,)
        policy_run = select_all(self.db, sql=sql, params=params)
        df = pd.DataFrame([[str(id), time, x] for id, time, l in policy_run for x in json.loads(l)], columns=['policy',
                                                                                                              'time',
                                                                                                              'reward'])
        # ToDo: Change this after run_baseline_policies
        if baseline:
            base = self._sim_baseline()
            size = max(len(json.loads(row[2])) for row in policy_run)
            df = df.append(pd.DataFrame([['baseline', '', base.run()] for _ in range(size)], columns=['policy',
                                                                                                      'time',
                                                                                                      'reward']))

        return df
