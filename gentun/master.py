#!/usr/bin/env python
"""
Master
"""

import pika
import Queue
import random
import threading
import uuid


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


if __name__ == '__main__':
    jobs = Queue.Queue()
    responses = Queue.Queue()
    for i in xrange(20):
        n = random.randint(25, 35)
        print(" [x] Requesting fib({})".format(n))
        jobs.put(True)
        client = RpcClient(jobs, responses)
        t = threading.Thread(target=client.call, args=[n])
        t.start()
    jobs.join()
    print [n for n in responses]
