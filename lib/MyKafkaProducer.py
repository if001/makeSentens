import sys
from kafka import KafkaProducer
from time import sleep
import json

class KafkaConst():
    PORT="9092"
    TOPIC='makesentens'
    IP1="35.194.228.2"
    IP2="45.76.199.23"


class MyKafkaProducer():
    def __init__(self):
        self.host1 = KafkaConst.IP1 + ':' + KafkaConst.PORT
        self.host2 = KafkaConst.IP2 + ':' + KafkaConst.PORT


    def create_producer(self):
        # 送信用のクライアントを作成
        try:
            self.producer = KafkaProducer(bootstrap_servers=[self.host1, self.host2],
                                          value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        except:
            print("kafka not operate")

    def send_message(self, x, y):
        try:
            data = {"x" : str(x), "y" : str(y)}
            print("send " + str(data) )
            self.producer.send(KafkaConst.TOPIC, data)
        except:
            print("kafka not operate")


def main():
    host1 = KafkaConst.IP1 + ':' + KafkaConst.PORT
    host2 = KafkaConst.IP2 + ':' + KafkaConst.PORT

    myKafkaProducer =  MyKafkaProducer()

    # 送信用のクライアントを作成
    myKafkaProducer.create_producer()

    message = sys.argv[-1]
    for i in range(10000):
        myKafkaProducer.send_message(i, i*2)
        sleep(2)

if __name__ == "__main__":
    main()
