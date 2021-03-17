import sys
from starlette.requests import Request
import ray
from ray import serve
from ray.serve.exceptions import RayServeException
from ray.serve.api import Client as ServeClient
from ray.serve import CondaEnv
import ray.rllib.agents.ppo as ppo
from utils import db_connect, BACKOFFICE_DB_NAME, TRAINER_DB_NAME, P_MARKER, select_record, SQLParamList, select_all, table_fetch_all
import json
import pandas as pd
from time import sleep
from datetime import datetime

from simpy_env import SimpyEnv
import subprocess
from configs.standard_azure_autoscaler import config
import os

_SHELL = os.getenv('SHELL')
_CONDA_PREFIX = os.getenv('CONDA_PREFIX_1') if 'CONDA_PREFIX_1' in os.environ.keys() else os.getenv('CONDA_PREFIX')

_BACKOFFICE_DB = db_connect(BACKOFFICE_DB_NAME)
_TRAINER_YAML = lambda trainer_name: "trainer_configs/{}_azure_scaler.yaml".format(trainer_name)
_TRAINER_PATH = lambda trainer_name: "trainer_"+trainer_name
_CMD_PREFIX = ". {}/etc/profile.d/conda.sh && conda activate simpy && ".format(_CONDA_PREFIX)


# ToDo: Add more exception handling
def launch_trainer(trainer_name: str = None):
    sql = '''SELECT data FROM trainer_cluster WHERE name = {}'''.format(P_MARKER)
    params = (trainer_name,)
    row = select_record(_BACKOFFICE_DB, sql=sql, params=params)
    if row is None:
        cursor = _BACKOFFICE_DB.cursor()
        cursor.execute('''INSERT INTO trainer_cluster (name) VALUES ({})'''.format(P_MARKER), (trainer_name,))
    #Check if training folder exists
    result = subprocess.run(['ls',_TRAINER_PATH(trainer_name)], capture_output=True, text=True)
    if result.returncode != 0:
        # Create trainer folder
        result = subprocess.run(['cp', '-r', 'trainer_template', _TRAINER_PATH(trainer_name)], capture_output=True,
                                text=True)
        if result.returncode:
            print("Error Creating Trainer Directory {}".format(_TRAINER_PATH(trainer_name)))
            print(result.stderr)
        # Create trainer yaml config file
    config_file = open(_TRAINER_YAML(trainer_name), "wt")
    config_file.write(config(trainer_name))
    config_file.close()
    # launch the cluster
    result = subprocess.run(_CMD_PREFIX + "ray up {} --no-config-cache -y".format(_TRAINER_YAML(
        trainer_name)), shell=True, capture_output=True, text=True, executable=_SHELL)
    _BACKOFFICE_DB.commit()
    return result

def tear_down_trainer(trainer_name: str = None):
    result = subprocess.run(_CMD_PREFIX + "ray down {} -y".format(_TRAINER_YAML(trainer_name)),
                            shell=True, capture_output=True, text=True, executable=_SHELL)
    return result

def get_trainer_data(trainer_name: str = None):
    result = subprocess.run(_CMD_PREFIX + "ray rsync_down {} '/home/ubuntu/trainer/' '{}'".format(
        _TRAINER_YAML(trainer_name), _TRAINER_PATH(trainer_name)),
                            shell=True, capture_output=True, text=True, executable=_SHELL)
    assert not result.returncode, \
        "Error on SyncDown {} {}\n{}".format(_TRAINER_YAML(trainer_name), _TRAINER_PATH(trainer_name), result.stderr)

    # Insert Update Policy Data from the Trainer DB
    # get the cluster id
    sql = '''SELECT id FROM trainer_cluster WHERE name = {}'''.format(P_MARKER)
    params = (trainer_name,)
    row = select_record(_BACKOFFICE_DB, sql=sql, params=params)
    assert row is not None, "Cluster {} does not Exist in {}.db".format(trainer_name,BACKOFFICE_DB_NAME)
    cluster_id, = row
    # get the policy data from the trainer db
    trainer_db = db_connect(_TRAINER_PATH(trainer_name)+"/"+TRAINER_DB_NAME)
    sql = '''SELECT policy.id, sim_model.name, policy.checkpoint, policy.agent_config
             FROM policy INNER JOIN sim_model ON policy.sim_model_id = sim_model.id'''
    cluster_policies = select_all(trainer_db, sql=sql)
    insert_sql = '''INSERT OR IGNORE INTO policy (
                        cluster_id,
                        policy_id,
                        model_name,
                        checkpoint,        
                        agent_config
                    ) VALUES ({})'''.format(SQLParamList(5))
    for policy_data in cluster_policies:
        cursor = _BACKOFFICE_DB.execute(insert_sql, (cluster_id,)+policy_data)

    _BACKOFFICE_DB.commit()

def get_policies():
    sql = '''SELECT policy.cluster_id as cluster_id, 
                    trainer_cluster.name as cluster_name, 
                    policy.model_name as model_name,
                    policy.checkpoint as checkpoint
             FROM policy INNER JOIN trainer_cluster ON policy.cluster_id = trainer_cluster.id'''
    return pd.read_sql_query(sql, _BACKOFFICE_DB)

# Removes backend services, and deletes local data
def remove_trainer(trainer_name: str = None):
    result = subprocess.run(['rm', '-r', _TRAINER_PATH(trainer_name)], capture_output=True, text=True)
    if result.returncode:
        print(result.stderr)
    result = subprocess.run(['rm', _TRAINER_YAML(trainer_name)], capture_output=True, text=True)
    if result.returncode:
        print(result.stderr)
    cursor = _BACKOFFICE_DB.cursor()
    sql = '''DELETE FROM trainer WHERE name = {}'''.format(P_MARKER)
    cursor.execute(sql, (trainer_name,))
    _BACKOFFICE_DB.commit()

def start_backend_server(address: str = None):
    stderrout = sys.stderr
    sys.stderr = open('modelserver.log', 'w')
    backend_id = None
    if not ray.is_initialized():
        if address is not None:
            ray.init(address=address)
        else:
            address = ray.init()
        try:
            backend_id = serve.start(detached=True)
            sleep(1)
        except RayServeException:
            backend_id = serve.connect()
    else:
        backend_id = serve.connect()

    sys.stderr = stderrout
    print("{} INFO Model Server started on {}".format(datetime.now(), address))
    print(
        "{} INFO Trainers Should Deploy Policies on this Server using address='{}'".format(datetime.now(), address))
    return backend_id


def deploy_policy(self, backend_server: ServeClient , trainer_id: int, policy_id: int, replicas: int = 1):
    class ServeModel:
        def __init__(self, agent_config: dict, checkpoint_path: str):

            # ToDo: Replace this after testing
            sim_name = 'trafic_light'
            exec_locals = {}
            try:
                exec(
                    "from models.{} import SimBaseline, N_ACTIONS, OBSERVATION_SPACE, SimModel, BASE_CONFIG".format(
                        sim_name), {},
                    exec_locals)
            except ModuleNotFoundError:
                raise Exception(" Model '{}' not found!!".format(sim_name))
            except Exception as e:
                raise e

            agent_config["env"] = SimpyEnv
            agent_config["env_config"] = {"n_actions"        : exec_locals['N_ACTIONS'],
                                          "observation_space": exec_locals['OBSERVATION_SPACE'],
                                          "sim_model"        : exec_locals['SimModel'],
                                          "sim_config"       : exec_locals['BASE_CONFIG']}
            print(agent_config)
            # assert agent_config is not None and isinstance(agent_config, dict), \
            #    "Invalid Agent Config {} when deploying a policy!".format(agent_config)
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

    sql = '''SELECT trainer_cluster.name, policy.checkpoint, policy.agent_config
             FROM policy INNER JOIN trainer_cluster ON policy.cluster_id = trainer_cluster.id
             WHERE cluster_id = {}, policy_id = {}'''.format(P_MARKER, P_MARKER)
    row = select_record(self.db, sql=sql, params=(trainer_id,policy_id))

    assert row is not None, "Invalid cluster_id {} and policy_id {}".format(trainer_id, policy_id)
    trainer_name, checkpoint, saved_agent_config = row
    saved_agent_config = json.loads(saved_agent_config)

    backend_name = "{}_policy_{}".format(trainer_name, policy_id)
    backend_server.create_backend(backend_name, ServeModel, saved_agent_config, checkpoint,
                                     config={'num_replicas': replicas}, env=CondaEnv("simpy"))
    print("# Backend Configured")
    route = "/{}".format(backend_name)
    backend_server.create_endpoint("{}_endpoint".format(backend_name), backend=backend_name, route=route)


#-------------------------------------------------- Old Code

# WARNING: This is not officially supported
def local_server_address():
    from ray._private.services import find_redis_address
    addresses = find_redis_address()
    assert len(addresses) == 1, "More than one Address Found {}".format(addresses)
    return addresses.pop()

def policy_id2str(model_name: str, policy_id: int):
    return "{}_policy_{}".format(model_name, policy_id)


class BackEndServer(ServeClient):
    def __init__(self, address: str = None, ):
        stderrout = sys.stderr
        sys.stderr = open('modelserver.log', 'w')
        if not ray.is_initialized():
            if address is not None:
                ray.init(address=address)
            else:
                address = ray.init()
            try:
                self.model_server = serve.start(detached=False)
                sleep(1)
            except RayServeException:
                self.model_server = serve.connect()
        else:
            self.model_server = serve.connect()

        sys.stderr = stderrout
        print("{} INFO Model Server started on {}".format(datetime.now(), address))
        print(
            "{} INFO Trainers Should Deploy Policies on this Server using address='{}'".format(datetime.now(), address))

        try:
            self.db = db_connect(BACKOFFICE_DB_NAME)
        except Exception as e:
            raise e

        self.trainer_path = lambda trainer_name: "trainer_" + trainer_name
        self.trainer_yaml = lambda trainer_name: "trainer_configs/{}_azure_scaler.yaml".format(trainer_name)
        self.activate_env = ". {}/etc/profile.d/conda.sh && conda activate simpy && ".format(os.getenv('CONDA_PREFIX'))


    # ToDo: Deploy a policy from a specific Trainer
    def deploy_policy(self, policy_id: int, replicas: int = 1):

        class ServeModel:
            def __init__(self, agent_config: dict, checkpoint_path: str):

                # ToDo: Replace this after testing
                sim_name = 'trafic_light'
                exec_locals = {}
                try:
                    exec(
                        "from models.{} import SimBaseline, N_ACTIONS, OBSERVATION_SPACE, SimModel, BASE_CONFIG".format(
                            sim_name), {},
                        exec_locals)
                except ModuleNotFoundError:
                    raise Exception(" Model '{}' not found!!".format(sim_name))
                except Exception as e:
                    raise e

                agent_config["env"] = SimpyEnv
                agent_config["env_config"] = {"n_actions"        : exec_locals['N_ACTIONS'],
                                              "observation_space": exec_locals['OBSERVATION_SPACE'],
                                              "sim_model"        : exec_locals['SimModel'],
                                              "sim_config"       : exec_locals['BASE_CONFIG']}
                print(agent_config)
                # assert agent_config is not None and isinstance(agent_config, dict), \
                #    "Invalid Agent Config {} when deploying a policy!".format(agent_config)
                print(checkpoint_path)
                # assert checkpoint_path is not None and isinstance(agent_config, str), \
                #    "Invalid Checkpoint Path {} when deploying a policy!".format(checkpoint_path)
                print("### 1")
                self.trainer = ppo.PPOTrainer(config=agent_config)
                print("### 2")
                self.trainer.restore(checkpoint_path)

            async def __call__(self, request: Request):
                json_input = await request.json()
                obs = json_input["observation"]

                action = self.trainer.compute_action(obs)
                return {"action": int(action)}

        select_policy_sql = '''SELECT sim_model.name, policy.checkpoint, policy.agent_config FROM policy
                               INNER JOIN sim_model ON policy.sim_model_id = sim_model.id
                               WHERE policy.id = {}'''.format(P_MARKER)
        row = select_record(self.db, sql=select_policy_sql, params=(policy_id,))

        assert row is not None, "Invalid Policy id {}".format(policy_id)
        model_name, checkpoint, saved_agent_config = row
        saved_agent_config = json.loads(saved_agent_config)

        if self.model_server is None:
            self.model_server = serve.connect()

        backend = policy_id2str(model_name, policy_id)
        print(saved_agent_config)
        print(checkpoint)
        self.model_server.create_backend(backend, ServeModel, saved_agent_config, checkpoint,
                                         config={'num_replicas': replicas}, env=CondaEnv("simpy"))
        print("# Backend Configured")
        route = "/{}".format(backend)
        self.model_server.create_endpoint("{}_endpoint".format(backend), backend=backend, route=route)

    # ToDo: Select policies from a Trainer
    def get_policies(self):
        sql = '''SELECT sim_model_id as model_id, sim_config_id as sim_config_id, id as policy_id
                 FROM policy '''
        return pd.read_sql_query(sql, self.db)


class CloudCluster:
    def __init__(self):
        pass
