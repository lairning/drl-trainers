# import sys
from starlette.requests import Request
import ray
from ray import serve
from ray.serve.exceptions import RayServeException
from ray.serve.api import Client as ServeClient
from ray.serve import CondaEnv
import ray.rllib.agents.ppo as ppo
from utils import db_connect, BACKOFFICE_DB_NAME, TRAINER_DB_NAME, P_MARKER, select_record, SQLParamList, select_all
import json
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
import os, sys

from simpy_template.simpy_env import SimpyEnv
from configs.scaler_config import trainer_cluster_config, server_cluster_config

_SHELL = os.getenv('SHELL')
_CONDA_PREFIX = os.getenv('CONDA_PREFIX_1') if 'CONDA_PREFIX_1' in os.environ.keys() else os.getenv('CONDA_PREFIX')

_BACKOFFICE_DB = db_connect(BACKOFFICE_DB_NAME)
_TRAINER_YAML = lambda cluster_name, cloud_provider: "configs/{}_{}_scaler.yaml".format(cluster_name, cloud_provider)
_TRAINER_PATH = lambda cluster_name, cloud_provider: "trainer_{}_{}".format(cluster_name, cloud_provider)
_CMD_PREFIX = ". {}/etc/profile.d/conda.sh && conda activate simpy && ".format(_CONDA_PREFIX)
_POLICY_SERVER_YAML = lambda cluster_name, cloud_provider: "configs/{}_{}_policy_server.yaml".format(cluster_name, cloud_provider)

_POLICY_ACTOR_CONFIG = {'num_cpus': 1}

_CURRENT_ENV = sys.executable.split('/')[-3]

def start_backend_server(config = None):
    #stderrout = sys.stderr
    #sys.stderr = open('modelserver.log', 'w')
    if not ray.is_initialized():
        ray.init(include_dashboard=False, log_to_driver=False, logging_level=0, address='auto')

    try:
        backend_server = serve.connect()
    except RayServeException:
        backend_server = serve.start(detached=True)

    if config != None:
        global _POLICY_ACTOR_CONFIG
        _POLICY_ACTOR_CONFIG = config

    #sys.stderr = stderrout
    #print("{} INFO Model Server started on {}".format(datetime.now(), addr))
    #print(
    #    "{} INFO Trainers Should Deploy Policies on this Server using address='{}'".format(datetime.now(), addr))
    return backend_server

def _get_trainer_and_cloud(trainer_id: int):
    sql = "SELECT name, cloud_provider FROM trainer_cluster WHERE id = {}".format(P_MARKER)
    row = select_record(_BACKOFFICE_DB, sql=sql, params=(trainer_id,))
    assert row is not None, "Unknown Trainer ID {}".format(trainer_id)
    return row

# ToDo: Add more exception handling
def launch_trainer(cluster_name: str = None, cloud_provider: str = '', cluster_config: dict = None):

    result = subprocess.run(['ls', _TRAINER_PATH(cluster_name, cloud_provider)], capture_output=True, text=True)
    # Create the Trainer Cluster if it does not exist.
    # No distinction exists between cloud providers, therefore training results are shared between runs in different
    # clouds
    if result.returncode != 0:
        # Create trainer folder
        result = subprocess.run(['cp', '-r', 'simpy_template', _TRAINER_PATH(cluster_name, cloud_provider)], capture_output=True,
                                text=True)
        if result.returncode:
            print("Error Creating Trainer Directory {}".format(_TRAINER_PATH(cluster_name, cloud_provider)))
            print(result.stderr)

        cursor = _BACKOFFICE_DB.cursor()
        sql = "INSERT INTO trainer_cluster (name, cloud_provider, start, config) VALUES ({})".format(SQLParamList(4))
        params = (cluster_name, cloud_provider, datetime.now(), json.dumps(cluster_config))
        cursor.execute(sql, params)
        _BACKOFFICE_DB.commit()
        trainer_id = cursor.lastrowid
    else:
        sql = '''SELECT id FROM trainer_cluster 
                 WHERE name = {} and cloud_provider = {} and stop IS NULL'''.format(P_MARKER, P_MARKER)
        trainer_id, = select_record(_BACKOFFICE_DB, sql=sql, params=(cluster_name, cloud_provider))
    # Create trainer yaml config file
    # When a cluster with the same name and provider is relaunched the configuration is overridden
    if cloud_provider != '':
        config_file = open(_TRAINER_YAML(cluster_name, cloud_provider), "wt")
        # ToDo: Test aws
        config_file.write(trainer_cluster_config(cloud_provider, cluster_name, _TRAINER_PATH(cluster_name, cloud_provider),
                                                 config=cluster_config))
        config_file.close()
        # launch the cluster
        result = subprocess.run(_CMD_PREFIX + "ray up {} --no-config-cache -y".format(_TRAINER_YAML(
            cluster_name, cloud_provider)), shell=True, capture_output=True, text=True, executable=_SHELL)
        subprocess.run(_CMD_PREFIX + "ray exec {} 'rm -r /home/ubuntu/trainer/*'".format(
            _TRAINER_YAML(cluster_name, cloud_provider)),
                       shell=True, capture_output=True, text=True, executable=_SHELL)
        subprocess.run(_CMD_PREFIX + "ray rsync_up {} '{}/' '/home/ubuntu/trainer/'".format(
            _TRAINER_YAML(cluster_name, cloud_provider), _TRAINER_PATH(cluster_name, cloud_provider)),
                       shell=True, capture_output=True, text=True, executable=_SHELL)

    _BACKOFFICE_DB.commit()
    return trainer_id, result

# ToDo: Test tear down of trainers already down. It seems it get stuck.
def tear_down_trainer(trainer_id: int):
    get_trainer_data(trainer_id=trainer_id)
    sql = "SELECT name, cloud_provider FROM trainer_cluster WHERE id = {}".format(P_MARKER)
    row = select_record(_BACKOFFICE_DB, sql=sql, params=(trainer_id,))
    assert row is not None, "Unknown Trainer ID {}".format(trainer_id)
    trainer_name, cloud_provider = row
    if cloud_provider != '':
        result = subprocess.run(_CMD_PREFIX + "ray down {} -y".format(_TRAINER_YAML(trainer_name, cloud_provider)),
                                shell=True, capture_output=True, text=True, executable=_SHELL)
        assert not result.returncode, "Error on Tear Down {} {}\n{}".format(_TRAINER_YAML(trainer_name, cloud_provider),
                                                                        _TRAINER_PATH(trainer_name, cloud_provider), result.stderr)

def monitor_trainer(trainer_id: int):

    trainer_name, cloud_provider = _get_trainer_and_cloud(trainer_id=trainer_id)
    if cloud_provider != '':
        result = subprocess.run(_CMD_PREFIX + "ray exec {} 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'".format(
            _TRAINER_YAML(trainer_name, cloud_provider)), shell=True, capture_output=True, text=True, executable=_SHELL)
        assert not result.returncode, "Error on SyncDown {} {}\n{}".format(_TRAINER_YAML(trainer_name, cloud_provider),
                                                                       _TRAINER_PATH(trainer_name, cloud_provider), result.stderr)

def get_trainer_data(trainer_id: int):

    trainer_name, cloud_provider = _get_trainer_and_cloud(trainer_id=trainer_id)
    if cloud_provider != '':
        result = subprocess.run(_CMD_PREFIX + "ray rsync_down {} '/home/ubuntu/trainer/' '{}'".format(
            _TRAINER_YAML(trainer_name, cloud_provider), _TRAINER_PATH(trainer_name, cloud_provider)),
                                shell=True, capture_output=True, text=True, executable=_SHELL)
        assert not result.returncode, "Error on SyncDown {} {}\n{}".format(_TRAINER_YAML(trainer_name, cloud_provider),
                                                                       _TRAINER_PATH(trainer_name, cloud_provider), result.stderr)

def delete_trainer(trainer_id: int):

    sql = '''SELECT count(*) FROM policy 
             WHERE trainer_id = {} AND backend_name IS NOT NULL'''.format(P_MARKER, P_MARKER)
    count, = select_record(_BACKOFFICE_DB, sql=sql, params=(trainer_id,))
    assert count == 0, "Can not delete trainer with deployed policies"
    tear_down_trainer(trainer_id=trainer_id)

    trainer_name, cloud_provider = _get_trainer_and_cloud(trainer_id=trainer_id)
    result = subprocess.run(['rm', '-r', _TRAINER_PATH(trainer_name, cloud_provider)], capture_output=True, text=True)
    if result.returncode:
        print(result.stderr)
    if cloud_provider != '':
        result = subprocess.run(['rm', _TRAINER_YAML(trainer_name, cloud_provider)], capture_output=True, text=True)
        if result.returncode:
                print(result.stderr)
    cursor = _BACKOFFICE_DB.cursor()
    sql = '''DELETE FROM trainer_cluster WHERE id = {}'''.format(P_MARKER)
    cursor.execute(sql, (trainer_id,))
    _BACKOFFICE_DB.commit()


def show_policies():
    sql = "SELECT id, name, cloud_provider FROM trainer_cluster"
    rows = select_all(_BACKOFFICE_DB, sql=sql)
    results = []
    policy_data_sql = '''SELECT sim_model.name as model_name, 
                    policy.sim_config_id as sim_config_id,
                    policy_run.policy_id as policy_id, 
                    policy_run.id as policy_run_id, 
                    policy_run.time_start as time_start, 
                    policy_run.simulations as simulations, 
                    policy_run.duration as duration, 
                    policy_run.results as results
             FROM policy_run 
             INNER JOIN policy ON policy_run.policy_id = policy.id
             INNER JOIN sim_model ON policy.sim_model_id = sim_model.id'''
    deployed_sql = '''SELECT policy_id
                           FROM policy WHERE trainer_id = {}'''.format(P_MARKER)
    for trainer_row in rows:
        trainer_id, trainer_name, cloud_provider = trainer_row
        trainer_db = db_connect(_TRAINER_PATH(trainer_name, cloud_provider) + "/" + TRAINER_DB_NAME)
        trainer_data = select_all(trainer_db,sql=policy_data_sql)
        deployed_policies = [row[0] for row in select_all(_BACKOFFICE_DB, sql=deployed_sql, params=(trainer_id,))]
        df_data = [trainer_row+data[:-1]+(np.mean(json.loads(data[-1])), np.std(json.loads(data[-1])),
                                          data[2] in deployed_policies) for data in trainer_data]
        df_columns = ['trainer_id', 'trainer_name', 'cloud_provider', 'model_name', 'sim_config_id', 'policy_id',
                      'run_id', 'time_start', 'simulations', 'duration', 'mean', 'std', 'deployed']
        results.append(pd.DataFrame(data=df_data, columns=df_columns))
    return pd.concat(results)


# ToDo: complete using the trainer.py code
def get_training_data(trainer_id: int, sim_config: int = None, baseline: bool = True):
    pass

# ToDo: complete using the trainer.py code
def get_policy_run_data(trainer_id: int, sim_config: int = None, baseline: bool = True):
    pass

def deploy_policy(backend_server: ServeClient, trainer_id: int, policy_id: int, policy_config: dict = None):
    class ServeModel:
        def __init__(self, agent_config: dict, checkpoint_path: str, trainer_path: str, model_name: str):

            sim_path = '{}.models.{}'.format(trainer_path, model_name)
            exec_locals = {}
            try:
                exec("from {} import SimBaseline, N_ACTIONS, OBSERVATION_SPACE, SimModel, BASE_CONFIG".format(
                    sim_path), {}, exec_locals)
            except ModuleNotFoundError:
                raise Exception(" Model '{}' not found!!".format(sim_path))
            except Exception as e:
                raise e

            agent_config["num_workers"] = 0
            agent_config["env"] = SimpyEnv
            agent_config["env_config"] = {"n_actions"        : exec_locals['N_ACTIONS'],
                                          "observation_space": exec_locals['OBSERVATION_SPACE'],
                                          "sim_model"        : exec_locals['SimModel'],
                                          "sim_config"       : exec_locals['BASE_CONFIG']}
            # print(agent_config)
            # assert agent_config is not None and isinstance(agent_config, dict), \
            #    "Invalid Agent Config {} when deploying a policy!".format(agent_config)
            checkpoint_path = trainer_path + checkpoint_path[1:]
            print(checkpoint_path)
            # assert checkpoint_path is not None and isinstance(agent_config, str), \
            #    "Invalid Checkpoint Path {} when deploying a policy!".format(checkpoint_path)
            self.trainer = ppo.PPOTrainer(config=agent_config)
            self.trainer.restore(checkpoint_path)

        async def __call__(self, request: Request):
            json_input = await request.json()
            obs = json_input["observation"]

            action = self.trainer.compute_action(obs)
            return {"action": int(action)}

    # Get Trainer DB
    trainer_name, cloud_provider = _get_trainer_and_cloud(trainer_id=trainer_id)
    trainer_db = db_connect(_TRAINER_PATH(trainer_name, cloud_provider) + "/" + TRAINER_DB_NAME)

    # Get Policy info
    sql = '''SELECT sim_model.name, policy.checkpoint, policy.agent_config
             FROM policy INNER JOIN sim_model ON policy.sim_model_id = sim_model.id
             WHERE policy.id = {}'''.format(P_MARKER)
    row = select_record(trainer_db, sql=sql, params=(policy_id,))
    assert row is not None, "Invalid Trainer ID {} and Policy ID {}".format(trainer_id, policy_id)
    model_name, checkpoint, saved_agent_config = row
    saved_agent_config = json.loads(saved_agent_config)

    if policy_config is None:
        policy_config = {'num_replicas': 1}
    policy_name = "trainer{}_policy{}".format(trainer_id, policy_id)
    trainer_path = _TRAINER_PATH(trainer_name, cloud_provider)
    backend_server.create_backend(policy_name, ServeModel, saved_agent_config, checkpoint, trainer_path, model_name,
                                  config=policy_config,
                                  ray_actor_options=_POLICY_ACTOR_CONFIG,
                                  env=CondaEnv(_CURRENT_ENV) )
    insert_sql = '''INSERT OR IGNORE INTO policy (
                        trainer_id,
                        policy_id,
                        backend_name
                    ) VALUES ({})'''.format(SQLParamList(3))
    cursor = _BACKOFFICE_DB.cursor()
    cursor.execute(insert_sql, (trainer_id, policy_id, policy_name))
    _BACKOFFICE_DB.commit()
    print("# Policy '{}' Deployed".format(policy_name))
    return policy_name

def undeploy_policy(backend_server: ServeClient, policy_name: str):
    sql = '''DELETE FROM policy 
             WHERE backend_name = {}'''.format(P_MARKER)
    cursor = _BACKOFFICE_DB.cursor()
    cursor.execute(sql, (policy_name, ))
    _BACKOFFICE_DB.commit()
    backend_server.delete_backend(policy_name)

def add_endpoint(backend_server: ServeClient, policy_name: str, endpoint_name: str):
    assert endpoint_name is not None and isinstance(endpoint_name,str), "Invalid endpoint {}".format(endpoint_name)
    endpoint_route = "/{}".format(endpoint_name)
    return backend_server.create_endpoint(endpoint_name, backend=policy_name, route=endpoint_route)

def delete_endpoint(backend_server: ServeClient, endpoint_name: str):
    return backend_server.delete_endpoint(endpoint_name)

def deploy_endpoint_policy(backend_server: ServeClient, trainer_id: int, policy_id: int, policy_config: dict = None,
                           endpoint_name: str = None):
    policy_name = deploy_policy(backend_server, trainer_id, policy_id, policy_config)
    if endpoint_name is None:
        endpoint_name = policy_name
    add_endpoint(backend_server,policy_name,endpoint_name)
    return endpoint_name, policy_name

def set_endpoint_traffic(backend_server: ServeClient, endpoint_name: str, traffic_config: dict):
    assert endpoint_name is not None and isinstance(endpoint_name,str), "Invalid endpoint {}".format(endpoint_name)
    assert traffic_config is not None and isinstance(traffic_config,dict), "Invalid endpoint {}".format(endpoint_name)
    backend_server.set_traffic(endpoint_name,traffic_config)

def get_simulator(trainer_id: int, policy_id: int):

    # Get Trainer DB
    trainer_name, cloud_provider = _get_trainer_and_cloud(trainer_id=trainer_id)
    trainer_db = db_connect(_TRAINER_PATH(trainer_name, cloud_provider) + "/" + TRAINER_DB_NAME)

    sql = '''SELECT trainer_cluster.name, trainer_cluster.cloud_provider, policy.model_name, policy.sim_config
             FROM policy INNER JOIN trainer_cluster ON policy.cluster_id = trainer_cluster.id
             WHERE policy_id = {}'''.format(P_MARKER, P_MARKER)
    sql = '''SELECT sim_model.name, sim_config.config
             FROM policy INNER JOIN sim_model ON policy.sim_model_id = sim_model.id
             INNER JOIN sim_config ON policy.sim_config_id = sim_config.id
             WHERE policy.id = {}'''.format(P_MARKER)
    row = select_record(trainer_db, sql=sql, params=(policy_id,))
    assert row is not None, "Invalid Trainer ID {} and Policy ID {}".format(trainer_id, policy_id)
    model_name, sim_config = row
    sim_config = json.loads(sim_config)

    sim_path = '{}.models.{}'.format(_TRAINER_PATH(trainer_name, cloud_provider), model_name)
    exec_locals = {}
    try:
        exec("from {} import SimBaseline, N_ACTIONS, OBSERVATION_SPACE, SimModel, BASE_CONFIG".format(
            sim_path), {}, exec_locals)
    except ModuleNotFoundError:
        raise Exception(" Model '{}' not found!!".format(sim_path))
    except Exception as e:
        raise e

    env_config = {"n_actions"        : exec_locals['N_ACTIONS'],
                  "observation_space": exec_locals['OBSERVATION_SPACE'],
                  "sim_model"        : exec_locals['SimModel'],
                  "sim_config"       : sim_config}

    return SimpyEnv(env_config)


def launch_policy_server(cluster_name: str = None, cloud_provider: str = '', cluster_config: dict = None):

    policy_server_yaml = _POLICY_SERVER_YAML(cluster_name, cloud_provider)

    result = subprocess.run(['ls', policy_server_yaml], capture_output=True, text=True)
    # Check if the cluster yaml file exists.
    if result.returncode != 0:

        config_file = open(policy_server_yaml, "wt")

        config_file.write(server_cluster_config(cloud_provider, cluster_name, config=cluster_config))
        config_file.close()
        # launch the cluster
        result = subprocess.run(_CMD_PREFIX + "ray up {} --no-config-cache -y".format(policy_server_yaml),
                                shell=True, capture_output=True, text=True, executable=_SHELL)
        subprocess.run(_CMD_PREFIX + "ray exec {} 'rm -r /home/ubuntu/server/*'".format(policy_server_yaml),
                       shell=True, capture_output=True, text=True, executable=_SHELL)
        subprocess.run(_CMD_PREFIX + "ray rsync_up {} './' '/home/ubuntu/server/'".format(policy_server_yaml),
                       shell=True, capture_output=True, text=True, executable=_SHELL)

    return result

def tear_down_policy_server(cluster_name: str = None, cloud_provider: str = ''):

    policy_server_yaml = _POLICY_SERVER_YAML(cluster_name, cloud_provider)

    result = subprocess.run(_CMD_PREFIX + "ray down {} -y".format(policy_server_yaml),
                            shell=True, capture_output=True, text=True, executable=_SHELL)
    subprocess.run(['rm', policy_server_yaml], capture_output=True, text=True)

    return result


def monitor_policy_server(cluster_name: str = None, cloud_provider: str = ''):

    policy_server_yaml = _POLICY_SERVER_YAML(cluster_name, cloud_provider)

    if cloud_provider != '':
        result = subprocess.run(_CMD_PREFIX + "ray exec {} 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'".format(
            policy_server_yaml), shell=True, capture_output=True, text=True, executable=_SHELL)
        assert not result.returncode, "Error on SyncDown {}\n{}".format(policy_server_yaml, result.stderr)







