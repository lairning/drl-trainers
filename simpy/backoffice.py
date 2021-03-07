import sys
from starlette.requests import Request
import ray
from ray import serve
from ray.serve.exceptions import RayServeException
from ray.serve import CondaEnv
import ray.rllib.agents.ppo as ppo
from utils import db_connect, DB_NAME, P_MARKER, select_record, SQLParamList, select_all
import json
import pandas as pd
from time import sleep
from datetime import datetime

# WARNING: This is not officially supported
def local_server_address():
    from ray._private.services import find_redis_address
    addresses = find_redis_address()
    assert len(addresses) == 1, "More than one Address Found {}".format(addresses)
    return addresses.pop()

def policy_id2str(policy_id:int):
    return "policy_{}".format(policy_id)

class ModelServer:
    def __init__(self, address:str=None, keep_alive: bool = False):
        stderrout = sys.stderr
        sys.stderr = open('modelserver.log', 'w')
        if not ray.is_initialized():
            if address is not None:
                ray.init(address=address)
            else:
                address = ray.init()
            try:
                self.model_server = serve.start(detached=keep_alive)
                sleep(1)
            except RayServeException:
                self.model_server = serve.connect()
        else:
            self.model_server = serve.connect()

        sys.stderr = stderrout
        print("{} INFO Model Server started on {}".format(datetime.now(),address))
        print("{} INFO Trainers Should Deploy Policies on this Server using address='{}'".format(datetime.now(),address))

        try:
            self.db = db_connect(DB_NAME)
        except Exception as e:
            raise e

    def list_backends(self):
        return self.model_server.list_backends()

    def delete_backend(self, policy_id: int):
        return self.model_server.delete_backend(backend_tag=policy_id2str(policy_id))

    def list_endpoints(self):
        return self.model_server.list_endpoints()

    def shutdown(self):
        self.model_server.shutdown()
        ray.shutdown()

    # ToDo: Deploy a policy from a specific Trainer
    def deploy_policy(self, policy_id: int, replicas: int = 1):

        class ServeModel:
            def __init__(self, agent_config: dict, checkpoint_path: str):
                assert agent_config is not None and isinstance(agent_config, dict), \
                    "Invalid Agent Config {} when deploying a policy!".format(agent_config)

                assert checkpoint_path is not None and isinstance(agent_config, str), \
                    "Invalid Checkpoint Path {} when deploying a policy!".format(checkpoint_path)
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

        if self.model_server is None:
            self.model_server = serve.connect()

        backend = policy_id2str(policy_id)
        print(saved_agent_config)
        print(checkpoint)
        self.model_server.create_backend(backend, ServeModel, saved_agent_config, checkpoint,
                                         config={'num_replicas': replicas})
        print("# Backend Configured")
        route = "{}".format(policy_id)
        self.model_server.create_endpoint("{}_endpoint".format(backend), backend=model_name, route=route)

    # ToDo: Select policies from a Trainer
    def get_policies(self):
        sql = '''SELECT sim_model_id as model_id, sim_config_id as sim_config_id, id as policy_id
                 FROM policy '''
        return pd.read_sql_query(sql, self.db)



class CloudCluster():
    def __init__(self):
        pass
