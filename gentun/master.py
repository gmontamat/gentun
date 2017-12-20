#!/usr/bin/env python
"""
Client to communicate with RabbitMQ and extensions of
individuals which add AMQP capabilities.
"""

import json
import pika
import Queue
import random
import threading
import uuid

from genetic_algorithm import Population


class RpcClient(object):
    """Define a client which sends work orders to a
    RabbitMQ message broker with a unique identifier
    and awaits for a response.
    """

    def __init__(self, jobs, responses, host='localhost', port=5672,
                 user='guest', password='guest', rabbit_queue='rpc_queue'):
        # Set connection and channel
        self.credentials = pika.PlainCredentials(user, password)
        self.parameters = pika.ConnectionParameters(host, port, '/', self.credentials)
        self.connection = pika.BlockingConnection(self.parameters)
        self.channel = self.connection.channel()
        # Set queue for jobs and callback queue for responses
        result = self.channel.queue_declare(exclusive=True)
        self.callback_queue = result.method.queue
        self.channel.basic_consume(self.on_response, no_ack=True, queue=self.callback_queue)
        self.rabbit_queue = rabbit_queue
        self.response = None
        self.id = None
        # Local queues shared between threads
        self.jobs = jobs
        self.responses = responses

    def on_response(self, channel, method, properties, body):
        if self.id == properties.correlation_id:
            self.response = body

    def call(self, parameters):
        self.id = str(uuid.uuid4())
        properties = pika.BasicProperties(reply_to=self.callback_queue, correlation_id=self.id)
        self.channel.basic_publish(
            exchange='', routing_key=self.rabbit_queue, properties=properties, body=str(parameters)
        )
        while self.response is None:
            self.connection.process_data_events()
        print(" [*] Got fib({})".format(parameters))
        self.responses.put(self.response)
        self.jobs.get()
        self.jobs.task_done()


class DistributedPopulation(Population):

    def __init__(self, species, individual_list=None, size=None,
                 uniform_rate=0.5, mutation_rate=0.015, additional_parameters=None):
        super(DistributedPopulation, self).__init__(
            species, None, None, individual_list, size, uniform_rate, mutation_rate, additional_parameters
        )

    def evaluate_in_parallel(self):
        """Send job requests to RabbitMQ pool so that workers
        evaluate individuals.
        """
        jobs = Queue.Queue()  # Acts as a counter of pending jobs
        responses = Queue.Queue()  # Collects fitness values from workers
        for i, individual in enumerate(self.individuals):
            if not individual.get_fitness_status():
                job_order = json.dumps([i, individual.get_genes(), individual.get_additional_parameters()])
                jobs.put(True)
                client = RpcClient(jobs, responses)
                thread = threading.Thread(target=client.call, args=[job_order])
                thread.start()
        jobs.join()  # Wait for all jobs to be completed
        # Collect results and assign to population
        while not responses.empty():
            response = responses.get(False)
            i, value = json.loads(response)
            self.individuals[i].set_fitness(value)


if __name__ == '__main__':
    pass
