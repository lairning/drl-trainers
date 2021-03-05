from starlette.requests import Request
import requests
from ray import serve
from ray.serve.exceptions import RayServeException

class ModelServer:
    def __init__(self):
        try:
            self.model_server = serve.connect()
        except RayServeException:
            self.model_server = serve.start(detached=True)
        except Exception as e:
            raise e

    def list_backends(self):
        return self.model_server.list_backends()

    def list_endpoints(self):
        return self.model_server.list_endpoints()