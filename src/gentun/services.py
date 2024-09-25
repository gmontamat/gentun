"""
This module handles job queueing and results retrieval using Redis.
"""

import inspect
import json
import logging
import os.path
import socket
import time
import uuid
from typing import Any, Optional, Type

import redis

from .models.base import Handler

CONTROLLER_MESSAGE = """

Fitness evaluation job sent to redis queue.
Use the following code on your worker nodes:

```
from gentun.models.{module} import {handler}
from gentun.services import RedisWorker

worker = RedisWorker("{name}", {handler}, host="{host}", port={port})
x_train, y_train, x_test, y_test = ...  # get data
worker.run(x_train, y_train, x_test, y_test)
```
"""


class RedisController:
    """Use redis as a queueing service to send jobs to workers."""

    def __init__(
        self,
        name: str,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        job_queue: str = "job_queue",
        results_queue: str = "results_queue",
        timeout: int = 259200,
    ):
        self.name = name
        self.host = host
        self.port = port
        self.client = redis.StrictRedis(host=host, port=port, password=password, db=0)
        self.job_queue = job_queue
        self.results_queue = results_queue
        self.timeout = timeout
        self.first_run = True

    def get_worker_details(self, handler: Type[Handler]) -> str:
        """Generate a sample code for worker."""
        if self.host in ["localhost", "127.0.0.1"]:
            connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            connection.connect(("8.8.8.8", 80))
            host = connection.getsockname()[0]
            connection.close()
        else:
            host = self.host
        module = os.path.basename(inspect.getfile(handler))[:-3]
        return CONTROLLER_MESSAGE.format(
            name=self.name, module=module, handler=handler.__name__, host=host, port=self.port
        )

    def send_job(self, handler: Type[Handler], **kwargs) -> str:
        """Send job data to the job queue for evaluation."""
        if self.first_run:
            logging.info(self.get_worker_details(handler))
            self.first_run = False
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "name": self.name,
            "handler": handler.__name__,
            "kwargs": kwargs,
        }
        self.client.lpush(self.job_queue, json.dumps(job))
        return job_id

    def wait_for_result(self, job_id) -> float:
        """Retrieve fitness from the results queue."""
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            data = self.client.rpop(self.results_queue)
            if data:
                result = json.loads(data)
                if result["name"] == self.name and result["id"] == job_id:
                    return result["fitness"]
                # Leave data back in queue
                self.client.lpush(self.results_queue, data)
            else:
                time.sleep(1)
        raise TimeoutError(f"Could not get job with id {job_id}")


class RedisWorker:
    """Read jobs from a redis and pass them to the model."""

    def __init__(
        self,
        name: str,
        handler: Type[Handler],
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        job_queue: str = "job_queue",
        results_queue: str = "results_queue",
        timeout: int = 259200,
    ):
        self.name = name
        self.handler = handler
        self.client = redis.StrictRedis(host=host, port=port, password=password, db=0)
        self.job_queue = job_queue
        self.results_queue = results_queue
        self.timeout = timeout

    def process_job(self, x_train: Any, y_train: Any, x_test: Any, y_test: Any, **kwargs) -> float:
        """Call model handler, return fitness."""
        return self.handler(**kwargs)(x_train, y_train, x_test, y_test)

    def run(self, x_train: Any, y_train: Any, x_test: Any = None, y_test: Any = None):
        """Read jobs from queue, call handler, and return fitness."""
        logging.info("Worker started (Ctrl+C to stop), waiting for jobs...")
        try:
            while True:
                job_data = self.client.rpop(self.job_queue)
                if job_data:
                    data = json.loads(job_data)
                    if data["name"] == self.name and data["handler"] == self.handler.__name__:
                        logging.info("Working on job %s", data["id"])
                        fitness = self.process_job(x_train, y_train, x_test, y_test, **data["kwargs"])
                        result = {"id": data["id"], "name": self.name, "fitness": fitness}
                        self.client.lpush(self.results_queue, json.dumps(result))
                    else:
                        # Job not used, do not dump
                        self.client.lpush(self.job_queue, job_data)
                else:
                    logging.debug("No jobs in queue, sleeping for a while...")
                    time.sleep(1)
        except KeyboardInterrupt:
            if job_data:
                self.client.lpush(self.job_queue, job_data)
            logging.info("Bye!")
