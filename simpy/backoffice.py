import sys
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
from datetime import datetime
import subprocess
import os

from simpy_template.simpy_env import SimpyEnv
from configs.scaler_config import azure_scaler_config

_SHELL = os.getenv('SHELL')
_CONDA_PREFIX = os.getenv('CONDA_PREFIX_1') if 'CONDA_PREFIX_1' in os.environ.keys() else os.getenv('CONDA_PREFIX')

_BACKOFFICE_DB = db_connect(BACKOFFICE_DB_NAME)
_TRAINER_YAML = lambda trainer_name, cloud_provider: "configs/{}_{}_scaler.yaml".format(trainer_name, cloud_provider)
_TRAINER_PATH = lambda trainer_name: "trainer_" + trainer_name
_CMD_PREFIX = ". {}/etc/profile.d/conda.sh && conda activate simpy && ".format(_CONDA_PREFIX)


def start_backend_server():
    stderrout = sys.stderr
    sys.stderr = open('modelserver.log', 'w')
    if not ray.is_initialized():
        ray.init(address='auto')

    try:
        backend_server = serve.connect()
    except RayServeException:
        backend_server = serve.start(detached=True)

    sys.stderr = stderrout
    print("{} INFO Model Server started on {}".format(datetime.now(), addr))
    print(
        "{} INFO Trainers Should Deploy Policies on this Server using address='{}'".format(datetime.now(), addr))
    return backend_server


# ToDo: Add more exception handling

def launch_trainer(trainer_name: str = None, cloud_provider: str = 'azure', config: dict = None):
    ''' Older Code
    sql = "SELECT data FROM trainer_cluster WHERE name = {}".format(P_MARKER)
    params = (trainer_name,)
    row = select_record(_BACKOFFICE_DB, sql=sql, params=params)
    if row is None:
        cursor = _BACKOFFICE_DB.cursor()
        cursor.execute("INSERT INTO trainer_cluster (name) VALUES ({})".format(P_MARKER), (trainer_name,))
    '''
    cursor = _BACKOFFICE_DB.cursor()
    sql = "INSERT INTO trainer_cluster (name, cloud_provider, start, config) VALUES ({})".format(SQLParamList(4))
    params = (trainer_name, cloud_provider, datetime.now(), json.dumps(config))
    cursor.execute(sql=sql, parameters=params)
    _BACKOFFICE_DB.commit()
    trainer_id = cursor.lastrowid
    #ToDo: Refactor to take into consideration the cloud vendor and the configurations
    # Check if training folder exists
    result = subprocess.run(['ls', _TRAINER_PATH(trainer_name)], capture_output=True, text=True)
    # Create the Trainer Cluster if it does not exist.
    # No distinction exists between cloud providers, therefore training results are shared between runs in different
    # clouds
    if result.returncode != 0:
        # Create trainer folder
        result = subprocess.run(['cp', '-r', 'simpy_template', _TRAINER_PATH(trainer_name)], capture_output=True,
                                text=True)
        if result.returncode:
            print("Error Creating Trainer Directory {}".format(_TRAINER_PATH(trainer_name)))
            print(result.stderr)
    # Create trainer yaml config file
    # When a cluster with the same name and provider is relaunched the configuration is overridden
    config_file = open(_TRAINER_YAML(trainer_name, cloud_provider), "wt")
    # ToDo: Add other Cloud Providers and use the
    config_file.write(azure_scaler_config(trainer_name))
    config_file.close()
    # launch the cluster
    result = subprocess.run(_CMD_PREFIX + "ray up {} --no-config-cache -y".format(_TRAINER_YAML(
        trainer_name, cloud_provider)), shell=True, capture_output=True, text=True, executable=_SHELL)
    _BACKOFFICE_DB.commit()
    return trainer_id, result

def tear_down_trainer(trainer_id: int):
    sql = "SELECT name, cloud_provider FROM trainer_cluster WHERE id = {}".format(P_MARKER)
    row = select_record(_BACKOFFICE_DB, sql=sql, params=(trainer_id,))
    assert row is not None, "Unknown Trainer ID {}".format(trainer_id)
    trainer_name, cloud_provider = row
    result = subprocess.run(_CMD_PREFIX + "ray down {} -y".format(_TRAINER_YAML(trainer_name)),
                            shell=True, capture_output=True, text=True, executable=_SHELL)
    assert not result.returncode, "Error on Tear Down {} {}\n{}".format(_TRAINER_YAML(trainer_name, cloud_provider),
                                                                        _TRAINER_PATH(trainer_name), result.stderr)
    sql = "UPDATE trainer_cluster SET stop = {} WHERE id = {}".format(P_MARKER, P_MARKER)
    cursor = _BACKOFFICE_DB.cursor()
    cursor.execute(sql=sql, parameters=(datetime.now(), trainer_id))
    _BACKOFFICE_DB.commit()
    return result


def get_trainer_data(trainer_id: int):
    sql = "SELECT name, cloud_provider FROM trainer_cluster WHERE id = {}".format(P_MARKER)
    row = select_record(_BACKOFFICE_DB, sql=sql, params=(trainer_id,))
    assert row is not None, "Unknown Trainer ID {}".format(trainer_id)
    trainer_name, cloud_provider = row
    result = subprocess.run(_CMD_PREFIX + "ray rsync_down {} '/home/ubuntu/trainer/' '{}'".format(
        _TRAINER_YAML(trainer_name, cloud_provider), _TRAINER_PATH(trainer_name)),
                            shell=True, capture_output=True, text=True, executable=_SHELL)
    assert not result.returncode, "Error on SyncDown {} {}\n{}".format(_TRAINER_YAML(trainer_name, cloud_provider),
                                                                       _TRAINER_PATH(trainer_name), result.stderr)
    # Insert Update Policy Data from the Trainer DB
    # get the cluster id
    sql = '''SELECT id FROM trainer_cluster WHERE name = {}'''.format(P_MARKER)
    params = (trainer_name,)
    row = select_record(_BACKOFFICE_DB, sql=sql, params=params)
    assert row is not None, "Cluster {} does not Exist in {}.db".format(trainer_name, BACKOFFICE_DB_NAME)
    cluster_id, = row
    # get the policy data from the trainer db
    trainer_db = db_connect(_TRAINER_PATH(trainer_name) + "/" + TRAINER_DB_NAME)
    sql = '''SELECT policy.id, sim_model.name, policy.checkpoint, policy.agent_config, sim_config.config
             FROM policy INNER JOIN sim_model ON policy.sim_model_id = sim_model.id
             INNER JOIN sim_config ON policy.sim_config_id = sim_config.id'''
    cluster_policies = select_all(trainer_db, sql=sql)
    insert_sql = '''INSERT OR IGNORE INTO policy (
                        cluster_id,
                        policy_id,
                        model_name,
                        checkpoint,        
                        agent_config,
                        sim_config
                    ) VALUES ({})'''.format(SQLParamList(6))
    for policy_data in cluster_policies:
        cursor = _BACKOFFICE_DB.execute(insert_sql, (trainer_id,) + policy_data)
    _BACKOFFICE_DB.commit()


def get_policies():
    sql = '''SELECT policy.cluster_id as trainer_id,
                    trainer_cluster.name as trainer_name,
                    policy.policy_id as policy_id, 
                    policy.model_name as model_name,
                    policy.checkpoint as checkpoint
             FROM policy INNER JOIN trainer_cluster ON policy.cluster_id = trainer_cluster.id'''
    return pd.read_sql_query(sql, _BACKOFFICE_DB)

#ToDo: Refactor to receive a trainer cluster id
# Removes backend services, and deletes local data
def remove_trainer(trainer_id: int):
    sql = "SELECT name, cloud_provider FROM trainer_cluster WHERE id = {}".format(P_MARKER)
    row = select_record(_BACKOFFICE_DB, sql=sql, params=(trainer_id,))
    assert row is not None, "Unknown Trainer ID {}".format(trainer_id)
    trainer_name, cloud_provider = row
    result = subprocess.run(['rm', '-r', _TRAINER_PATH(trainer_name)], capture_output=True, text=True)
    if result.returncode:
        print(result.stderr)
    result = subprocess.run(['rm', _TRAINER_YAML(trainer_name, cloud_provider)], capture_output=True, text=True)
    if result.returncode:
        print(result.stderr)
    cursor = _BACKOFFICE_DB.cursor()
    sql = '''DELETE FROM trainer_cluster WHERE id = {}'''.format(P_MARKER)
    cursor.execute(sql, (trainer_id,))
    _BACKOFFICE_DB.commit()

def add_policy(backend_server: ServeClient, trainer_id: int, policy_id: int, policy_config: dict = None):
    class ServeModel:
        def __init__(self, agent_config: dict, checkpoint_path: str, trainer_name: str, model_name: str):

            sim_path = '{}.models.{}'.format(_TRAINER_PATH(trainer_name), model_name)
            exec_locals = {}
            try:
                exec("from {} import SimBaseline, N_ACTIONS, OBSERVATION_SPACE, SimModel, BASE_CONFIG".format(
                    sim_path), {}, exec_locals)
            except ModuleNotFoundError:
                raise Exception(" Model '{}' not found!!".format(sim_path))
            except Exception as e:
                raise e

            agent_config["env"] = SimpyEnv
            agent_config["env_config"] = {"n_actions"        : exec_locals['N_ACTIONS'],
                                          "observation_space": exec_locals['OBSERVATION_SPACE'],
                                          "sim_model"        : exec_locals['SimModel'],
                                          "sim_config"       : exec_locals['BASE_CONFIG']}
            # print(agent_config)
            # assert agent_config is not None and isinstance(agent_config, dict), \
            #    "Invalid Agent Config {} when deploying a policy!".format(agent_config)
            checkpoint_path = _TRAINER_PATH(trainer_name) + checkpoint_path[1:]
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

    sql = '''SELECT trainer_cluster.name, policy.model_name, policy.checkpoint, policy.agent_config
             FROM policy INNER JOIN trainer_cluster ON policy.cluster_id = trainer_cluster.id
             WHERE cluster_id = {} AND policy_id = {}'''.format(P_MARKER, P_MARKER)
    row = select_record(_BACKOFFICE_DB, sql=sql, params=(trainer_id, policy_id))

    assert row is not None, "Invalid cluster_id {} and policy_id {}".format(trainer_id, policy_id)
    trainer_name, model_name, checkpoint, saved_agent_config = row
    saved_agent_config = json.loads(saved_agent_config)

    policy_name = "{}_{}".format(trainer_name, policy_id)
    if policy_config is None:
        policy_config = {'num_replicas': 1}
    backend_server.create_backend(policy_name, ServeModel, saved_agent_config, checkpoint, trainer_name, model_name,
                                  config=policy_config, env=CondaEnv("simpy"))
    print("# Policy '{}' Configured".format(policy_name))
    return policy_name

def add_endpoint(backend_server: ServeClient, policy_name: str, endpoint_name: str):
    assert endpoint_name is not None and isinstance(endpoint_name,str), "Invalid endpoint {}".format(endpoint_name)
    endpoint_route = "/{}".format(endpoint_name)
    backend_server.create_endpoint(endpoint_name, backend=policy_name, route=endpoint_route)

def deploy_policy(backend_server: ServeClient, trainer_id: int, policy_id: int, policy_config: dict = None,
                  endpoint_name: str = None):
    policy_name = add_policy(backend_server, trainer_id, policy_id, policy_config)
    if endpoint_name is None:
        sql = '''SELECT model_name
                 FROM policy
                 WHERE cluster_id = {} AND policy_id = {}'''.format(P_MARKER, P_MARKER)
        endpoint_name, = select_record(_BACKOFFICE_DB, sql=sql, params=(trainer_id, policy_id))
    add_endpoint(backend_server,policy_name,endpoint_name)
    return endpoint_name, policy_name

def set_endpoint_traffic(backend_server: ServeClient, endpoint_name: str, traffic_config: dict):
    assert endpoint_name is not None and isinstance(endpoint_name,str), "Invalid endpoint {}".format(endpoint_name)
    assert traffic_config is not None and isinstance(traffic_config,dict), "Invalid endpoint {}".format(endpoint_name)
    backend_server.set_traffic(endpoint_name,traffic_config)

def get_simulator(trainer_id: int, policy_id: int):
    sql = '''SELECT trainer_cluster.name, policy.model_name, policy.sim_config
             FROM policy INNER JOIN trainer_cluster ON policy.cluster_id = trainer_cluster.id
             WHERE cluster_id = {} AND policy_id = {}'''.format(P_MARKER, P_MARKER)
    row = select_record(_BACKOFFICE_DB, sql=sql, params=(trainer_id, policy_id))

    assert row is not None, "Invalid cluster_id {} and policy_id {}".format(trainer_id, policy_id)
    trainer_name, model_name, sim_config = row
    sim_config = json.loads(sim_config)

    sim_path = '{}.models.{}'.format(_TRAINER_PATH(trainer_name), model_name)
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














