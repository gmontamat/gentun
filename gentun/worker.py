#!/usr/bin/env python
"""
Extend models by turning them into AMPQ consumers
"""

import pika


def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


class GentunWorker(object):

    def __init__(self, model):
        self.model = model
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')

    def on_request(self, channel, method, properties, body):
        n = int(body)
        print(" [.] fib(%s)" % n)
        response = self.model(n)
        channel.basic_publish(
            exchange='', routing_key=properties.reply_to,
            properties=pika.BasicProperties(correlation_id=properties.correlation_id), body=str(response)
        )
        channel.basic_ack(delivery_tag=method.delivery_tag)

    def work(self):
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(self.on_request, queue='rpc_queue')
        print(" [x] Awaiting master requests")
        self.channel.start_consuming()


if __name__ == '__main__':
    gw = GentunWorker(fib)
    gw.work()
