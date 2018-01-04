#!/usr/bin/env python
"""
Define worker class which loads a train set and receives
job orders from a master via a RabbitMQ message broker.
"""

import json
import pika
import random
import threading
import time


def sample_model():
    """Emulate model cross-validation for debugging purposes."""
    time.sleep(random.randint(1, 10))
    return random.random()


class GentunWorker(object):

    def __init__(self, model, x_train, y_train, host='localhost', port=5672,
                 user='guest', password='guest', rabbit_queue='rpc_queue'):
        self.model = model
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
        print(" [.] Evaluating individual {}".format(i))
        # print("     ... Genes: {}".format(str(genes)))
        # print("     ... Other: {}".format(str(additional_parameters)))
        # Run model and return cross-validation metric
        model = self.model(self.x_train, self.y_train, genes, **additional_parameters)
        fitness = model.cross_validate()
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
            self.channel.basic_consume(self.on_request, queue=self.rabbit_queue)
            print(" [x] Awaiting master's requests")
            print(" [-] Press Ctrl+C to interrupt")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print("\nGood bye!")


if __name__ == '__main__':
    import pandas as pd
    from models import XgboostModel

    data = pd.read_csv('../tests/wine-quality/winequality-white.csv', delimiter=';')
    y = data['quality']
    x = data.drop(['quality'], axis=1)
    gw = GentunWorker(XgboostModel, x, y)
    gw.work()
