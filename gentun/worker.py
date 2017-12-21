#!/usr/bin/env python
"""
Worker
"""

import json
import pika
import random
import time


def sample_model():
    time.sleep(random.randint(3, 7))
    return random.random()


class GentunWorker(object):

    def __init__(self, model):
        self.model = model
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')

    def on_request(self, channel, method, properties, body):
        i, genes, additional_parameters = json.loads(body)
        print(" [.] Evaluating individual {}".format(i))
        print("     ... Genes: {}".format(str(genes)))
        print("     ... Other: {}".format(str(additional_parameters)))
        fitness = self.model()
        response = json.dumps([i, fitness])
        channel.basic_publish(
            exchange='', routing_key=properties.reply_to,
            properties=pika.BasicProperties(correlation_id=properties.correlation_id), body=response
        )
        channel.basic_ack(delivery_tag=method.delivery_tag)

    def work(self):
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(self.on_request, queue='rpc_queue')
        print(" [x] Awaiting master requests")
        self.channel.start_consuming()


if __name__ == '__main__':
    gw = GentunWorker(sample_model)
    gw.work()
