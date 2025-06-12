# producer.py
from kafka import KafkaProducer
import json
import time
from sklearn.datasets import load_iris
import random

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Load Iris dataset
iris = load_iris()
X = iris.data[:, :3]  # Only 3 features used during training

# Send data to Kafka topic
while True:
    idx = random.randint(0, len(X) - 1)
    data_point = X[idx].tolist()
    print(f"Sending data: {data_point}")
    producer.send('iris-topic', value=data_point)
    time.sleep(1)
