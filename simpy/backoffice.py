from starlette.requests import Request
import requests
from ray import serve


class ModelServer:
    def __init__(self):
        self.model_server = serve.connect()

    def list_backends(self):
        return self.model_server.list_backends()

    def list_endpoints(self):
        return self.model_server.list_endpoints()