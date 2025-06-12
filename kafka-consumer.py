# consumer.py
from kafka import KafkaConsumer
import json
import joblib
import numpy as np

# Load pre-trained model
model = joblib.load("model.pkl")  # Make sure this file exists

# Initialize Kafka Consumer
consumer = KafkaConsumer(
    'iris-topic',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print("Consumer started, waiting for messages...")

# Start consuming
for message in consumer:
    features = np.array(message.value).reshape(1, -1)
    prediction = model.predict(features)[0]
    print(f"Received: {message.value} => Prediction: {prediction}")
