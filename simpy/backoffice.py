from starlette.requests import Request
import requests
from ray import serve
from ray.serve.exceptions import RayServeException
import ray.rllib.agents.ppo as ppo
from utils import db_connect, DB_NAME, P_MARKER, select_record, SQLParamList, select_all
import json
import pandas as pd

class ModelServer:
    def __init__(self):
        try:
            self.model_server = serve.connect()
        except RayServeException:
            self.model_server = serve.start(detached=True)
        except Exception as e:
            raise e

        try:
            self.db = db_connect(DB_NAME)
        except Exception as e:
            raise e

    def list_backends(self):
        return self.model_server.list_backends()

    def list_endpoints(self):
        return self.model_server.list_endpoints()

    def deploy_policy(self, policy_id: int, replicas: int = 1):

        class ServeModel:
            def __init__(self, agent_config: dict, checkpoint_path: str):
                assert agent_config is not None and isinstance(agent_config, dict), \
                    "Invalid Agent Config {} when deploying a policy!".format(agent_config)

                assert checkpoint_path is not None and isinstance(agent_config, str), \
                    "Invalid Checkpoint Path {} when deploying a policy!".format(checkpoint_path)

                self.trainer = ppo.PPOTrainer(config=agent_config)
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

        agent_config = self._config.copy()
        agent_config.update(json.loads(saved_agent_config))

        if self.model_server is None:
            self.model_server = serve.connect()

        backend = "policy_{}".format(policy_id)
        self.model_server.create_backend(backend, ServeModel, agent_config, checkpoint,
                                         config={'num_replicas': replicas})
        route = "{}/{}".format(model_name, policy_id)
        self.model_server.create_endpoint("{}_endpoint".format(backend), backend=model_name, route=route)

    def get_policies(self):
        sql = '''SELECT sim_config_id as sim_config, id as policy, session_id as session
                 FROM policy
                 WHERE sim_model_id = {}'''.format(P_MARKER)
        return pd.read_sql_query(sql, self.db, params=(self._model_id,))
