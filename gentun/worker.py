#!/usr/bin/env python
"""
Worker
"""

import json
import pika
import random
import time

from models import XgboostRegressor


def sample_model():
    time.sleep(random.randint(3, 7))
    return random.random()


class GentunWorker(object):

    def __init__(self, model, x_train, y_train):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')

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
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(self.on_request, queue='rpc_queue')
        print(" [x] Awaiting master's requests")
        self.channel.start_consuming()


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv('../tests/wine-quality/winequality-white.csv', delimiter=';')
    y = data['quality']
    x = data.drop(['quality'], axis=1)
    gw = GentunWorker(XgboostRegressor, x, y)
    gw.work()
