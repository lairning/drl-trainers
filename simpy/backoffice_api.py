
import ray
from ray import serve
from ray.serve.exceptions import RayServeException
from ray.serve.api import Client as ServeClient
from ray.serve import CondaEnv

import pandas as pd

from utils import db_connect, BACKOFFICE_DB_NAME, TRAINER_DB_NAME, P_MARKER, select_record, SQLParamList, select_all

# API Services

def get_policies_api(request):
    sql = '''SELECT policy.cluster_id as trainer_id,
                    trainer_cluster.name as trainer_name,
                    policy.policy_id as policy_id, 
                    policy.model_name as model_name,
                    policy.checkpoint as checkpoint
             FROM policy INNER JOIN trainer_cluster ON policy.cluster_id = trainer_cluster.id'''
    db = db_connect(BACKOFFICE_DB_NAME, check_same_thread=False)
    df = pd.read_sql_query(sql, db)
    return df.to_json()

# Each endpoint is defined by the backend name, function and http route
BACKOFFICE_ENDPOINTS = [('get_policies',get_policies_api,'/policies')]

# Start Backend

if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init(address='auto')

    try:
        backend_server = serve.connect()
    except RayServeException:
        backend_server = serve.start(detached=True)

    backend_list = list(backend_server.list_backends().keys())

    policy_config = {'num_replicas': 1}

    for backend, service_function, route in BACKOFFICE_ENDPOINTS:
        if backend not in backend_list:
            backend_server.create_backend(backend, service_function, config=policy_config, env=CondaEnv("simpy"))
            backend_server.create_endpoint(backend, backend=backend, route=route)