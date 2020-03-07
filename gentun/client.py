#!/usr/bin/env python
"""
Define client class which loads a train set and receives
job orders from a master via a RabbitMQ message broker.
"""

import json
import pika
import threading
import time


class GentunClient(object):

    def __init__(self, individual, x_train, y_train, host='localhost', port=5672,
                 user='guest', password='guest', rabbit_queue='rpc_queue'):
        self.individual = individual
        self.x_train = x_train
        self.y_train = y_train
        self.credentials = pika.PlainCredentials(user, password)
        self.parameters = pika.ConnectionParameters(host, port, '/', self.credentials)
        self.connection = pika.BlockingConnection(self.parameters)
        self.channel = self.connection.channel()
        self.rabbit_queue = rabbit_queue
        self.channel.queue_declare(queue=self.rabbit_queue)
        # Report to the RabbitMQ server
        heartbeat_thread = threading.Thread(target=self.heartbeat)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()

    def heartbeat(self):
        """Send heartbeat messages to RabbitMQ server."""
        while True:
            time.sleep(10)
            self.connection.process_data_events()

    def on_request(self, channel, method, properties, body):
        i, genes, additional_parameters = json.loads(body)
        # If an additional parameter is received as a list, convert to tuple
        for param in additional_parameters.keys():
            if isinstance(additional_parameters[param], list):
                additional_parameters[param] = tuple(additional_parameters[param])
        print(" [.] Evaluating individual {}".format(i))
        # print("     ... Genes: {}".format(str(genes)))
        # print("     ... Other: {}".format(str(additional_parameters)))
        # Run model and return fitness metric
        individual = self.individual(self.x_train, self.y_train, genes=genes, **additional_parameters)
        fitness = individual.get_fitness()
        # Prepare response for master and send it
        response = json.dumps([i, fitness])
        channel.basic_publish(
            exchange='', routing_key=properties.reply_to,
            properties=pika.BasicProperties(correlation_id=properties.correlation_id), body=response
        )
        channel.basic_ack(delivery_tag=method.delivery_tag)

    def work(self):
        try:
            self.channel.basic_qos(prefetch_count=1)
            self.channel.basic_consume(queue=self.rabbit_queue, on_message_callback=self.on_request)
            print(" [x] Awaiting master's requests")
            print(" [-] Press Ctrl+C to interrupt")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print()
            print("Good bye!")
