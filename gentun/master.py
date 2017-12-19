#!/usr/bin/env python
"""
Extend individuals by turning them into AMPQ publishers
"""

import pika
import uuid


class RpcClient(object):

    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')
        )
        self.channel = self.connection.channel()
        result = self.channel.queue_declare(exclusive=True)
        self.callback_queue = result.method.queue
        self.channel.basic_consume(self.on_response, no_ack=True, queue=self.callback_queue)

    def on_response(self, channel, method, properties, body):
        if self.corr_id == properties.correlation_id:
            self.response = body

    def call(self, n):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='', routing_key='rpc_queue', properties=pika.BasicProperties(
                reply_to=self.callback_queue, correlation_id=self.corr_id,
            ), body=str(n)
        )
        while self.response is None:
            self.connection.process_data_events()
        return float(self.response)


import random
import threading

responses = []
for i in xrange(20):
    n = random.randint(25, 35)
    print(" [x] Requesting fib({})".format(n))
    rpc_client = RpcClient()
    t = threading.Thread(target=rpc_client.call, args=[n])
    t.start()
