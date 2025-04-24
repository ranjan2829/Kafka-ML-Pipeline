from kafka import kafkaProducer
import json
import time
producer=kafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
)
data = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
while True:
    producer.send("iris_topic", data)
    print("Sent:", data)
    time.sleep(2)
