#!/usr/bin/env python
"""
Client to communicate with RabbitMQ and extension of
Population which add parallel computing capabilities.
"""

import json
import pika
import Queue
import threading
import time
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
        # Report to the RabbitMQ server
        heartbeat_thread = threading.Thread(target=self.heartbeat)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()

    def heartbeat(self):
        """Send heartbeat messages to RabbitMQ server."""
        while True:
            time.sleep(10)
            try:
                self.connection.process_data_events()
            except pika.exceptions.ConnectionClosed:
                # Connection was closed, stop sending heartbeat messages
                break

    def on_response(self, channel, method, properties, body):
        if self.id == properties.correlation_id:
            self.response = body

    def call(self, parameters):
        assert type(parameters) == str
        self.id = str(uuid.uuid4())
        properties = pika.BasicProperties(reply_to=self.callback_queue, correlation_id=self.id)
        self.channel.basic_publish(
            exchange='', routing_key=self.rabbit_queue, properties=properties, body=parameters
        )
        while self.response is None:
            time.sleep(3)
        print(" [*] Got fitness for individual {}".format(json.loads(parameters)[0]))
        self.responses.put(self.response)
        # Job is completed, remove job order from queue
        self.jobs.get()
        self.jobs.task_done()
        # Close RabbitMQ connection to prevent file descriptors limit from blocking server
        self.connection.close()


class DistributedPopulation(Population):
    """Override Population class by making x_train and
    y_train optional parameters set to None and sending
    evaluation requests to the workers before computing
    the fittest individual.
    """

    def __init__(self, species, x_train=None, y_train=None, individual_list=None, size=None,
                 uniform_rate=0.5, mutation_rate=0.015, additional_parameters=None,
                 host='localhost', port=5672, user='guest', password='guest', rabbit_queue='rpc_queue'):
        assert x_train is None and y_train is None  # Train set is not necessary since workers have it
        super(DistributedPopulation, self).__init__(
            species, x_train, y_train, individual_list, size, uniform_rate, mutation_rate, additional_parameters
        )
        self.credentials = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'rabbit_queue': rabbit_queue
        }

    def get_fittest(self):
        """Evaluate necessary individuals in parallel before getting fittest."""
        self.evaluate_in_parallel()
        return super(DistributedPopulation, self).get_fittest()

    def evaluate_in_parallel(self):
        """Send job requests to RabbitMQ pool so that
        workers evaluate individuals with unknown fitness.
        """
        jobs = Queue.Queue()  # "Counter" of pending jobs, shared between threads
        responses = Queue.Queue()  # Collect fitness values from workers
        for i, individual in enumerate(self.individuals):
            if not individual.get_fitness_status():
                job_order = json.dumps([i, individual.get_genes(), individual.get_additional_parameters()])
                jobs.put(True)
                client = RpcClient(jobs, responses, **self.credentials)
                communication_thread = threading.Thread(target=client.call, args=[job_order])
                communication_thread.daemon = True
                communication_thread.start()
        jobs.join()  # Block here until all jobs are completed
        # Collect results and assign them to their respective individuals
        while not responses.empty():
            response = responses.get(False)
            i, value = json.loads(response)
            self.individuals[i].set_fitness(value)


if __name__ == '__main__':
    from genetic_algorithm import GeneticAlgorithm
    from individuals import XgboostIndividual

    pop = DistributedPopulation(XgboostIndividual, size=100, additional_parameters={'nfold': 3})
    ga = GeneticAlgorithm(pop)
    ga.run(10)
