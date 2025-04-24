from kafka import KafkaConsumer
import json
import pickle
import numpy as np

with open('model.pkl','rb') as f:
    model = pickle.load(f)
consumer=KafkaConsumer(
    "iris_topic",
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
)
for msg in consumer:
    data=msg.value
    features=np.array([[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]])
    prediction=model.predict(features)
    print("Received:", data)    
    print("Predicted class:", prediction[0])