
import ray
from ray import serve
from ray.serve.exceptions import RayServeException
from ray.serve.api import Client as ServeClient
from ray.serve import CondaEnv

import pandas as pd
import json
from utils import db_connect, BACKOFFICE_DB_NAME, TRAINER_DB_NAME, P_MARKER, select_record, SQLParamList, select_all


def my_json_load(data):
    if data is None:
        return None
    return json.loads(data)

# API Services

#ToDo:  recreate backoffice.db trainer_cluster table to add more trainer properties
#       change the backoffice launch_trainer function to add the new trainer properties
#       resync trainer data
#ToDo:  Add a table trainer Cluster runs to the backoffice.db to store periods the cluster was up
#       change the backoffice launch_trainer function and tear down to add records to the new table
#       resync trainer data

def get_trainers_api(request):
    sql = '''SELECT id, name, cloud_provider, start, stop, config FROM trainer_cluster'''
    db = db_connect(BACKOFFICE_DB_NAME, check_same_thread=False)
    rows = select_all(db, sql)
    data = [(i, name, cloud_provider, start, stop, json.loads(config))
            for i, name, cloud_provider, start, stop, config in rows]
    return json.dumps(data)

def get_endpoints_api(request):
    backend = serve.connect()
    data = {k:v for k,v in backend.list_endpoints().items() if k not in BACKOFFICE_ENDPOINTS.keys()}
    return json.dumps(data)

def get_policies_api(request):
    def _process(row, backend_list, backend_traffic):
        row[5] = json.loads(row[5])
        row[6] = json.loads(row[6])
        row += (None,None) if row[7] is None else (backend_list[row[7]].num_replicas,
                                                   backend_list[row[7]].max_concurrent_queries)
        row += (None,) if row[7] is None else (backend_traffic.get(row[7],None),)
        return tuple(row)

    sql = '''SELECT policy.cluster_id as trainer_id,
                    trainer_cluster.name as trainer_name,
                    policy.policy_id as policy_id, 
                    policy.model_name as model_name,
                    policy.checkpoint as checkpoint,
                    policy.agent_config,
                    policy.sim_config,
                    policy.backend_name
             FROM policy INNER JOIN trainer_cluster ON policy.cluster_id = trainer_cluster.id'''
    db = db_connect(BACKOFFICE_DB_NAME, check_same_thread=False)
    rows = select_all(db, sql)
    backend = serve.connect()
    backend_traffic = { backend:endpoint for endpoint,props in backend.list_endpoints().items()
                            if endpoint not in BACKOFFICE_ENDPOINTS.keys()
                            for backend in props['traffic'].keys()}
    data = [_process(list(row), backend.list_backends(), backend_traffic) for row in rows]
    return json.dumps(data)


# Each endpoint is defined by the backend name, function and http route
BACKOFFICE_ENDPOINTS = {'get_trainers':(get_trainers_api,'/trainers'),
                        'get_endpoints': (get_endpoints_api, '/endpoints'),
                        'get_policies': (get_policies_api, '/policies'),
                        }

# Start Backend

if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init(address='auto')

    try:
        backend_server = serve.connect()
    except RayServeException:
        backend_server = serve.start(detached=True)

    endpoint_list = list(backend_server.list_endpoints().keys())
    backend_list = list(backend_server.list_backends().keys())

    policy_config = {'num_replicas': 1}

    for name, (service_function, route) in BACKOFFICE_ENDPOINTS.items():
        if name in endpoint_list:
            backend_server.delete_endpoint(name)
        if name in backend_list:
            backend_server.delete_backend(name)
        backend_server.create_backend(name, service_function, config=policy_config, env=CondaEnv("simpy"))
        backend_server.create_endpoint(name, backend=name, route=route)