from confluent_kafka import Producer
import json
import time
import random
config = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'ai-text-producer'
}
producer = Producer(config)
def delivery_report(err, msg):
    """Called once for each message produced to indicate delivery result."""
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')
creative_prompts = [
    "The mystical forest",
    "In the digital realm",
    "Time flows like",
    "Artificial intelligence learns",
    "The quantum world",
    "Deep learning networks",
    "Stories have the power",
    "Technology bridges gaps",
    "Neural networks discover",
    "The future holds",
    "Machine learning algorithms",
    "Creativity emerges from",
    "Data scientists work",
    "Innovation happens when",
    "The digital age",
    "Generative models create",
    "Transformers revolutionized",
    "Deep neural networks",
    "Once upon a time",
    "In a world where",
    "The ancient wizard",
    "Through the cosmic void",
    "Beyond the horizon",
    "In the depths of",
    "The mechanical heart",
    "Across dimensions of",
    "Within the code",
    "The intelligent machine",
    "Dreams of tomorrow",
    "In the language of"
]
generation_settings = [
    {"max_length": 20, "temperature": 0.7, "top_k": 50},
    {"max_length": 25, "temperature": 0.8, "top_k": 40},
    {"max_length": 30, "temperature": 0.9, "top_k": 30},
    {"max_length": 15, "temperature": 0.6, "top_k": 60},
    {"max_length": 35, "temperature": 1.0, "top_k": 25}
]
try:
    while True:
        prompt = random.choice(creative_prompts)
        settings = random.choice(generation_settings)
        message_data = {
            "prompt": prompt,
            "max_length": settings["max_length"],
            "temperature": settings["temperature"],
            "top_k": settings["top_k"],
            "timestamp": time.time()
        }
        print(f"Sending prompt: '{prompt}' with settings: {settings}")
        producer.produce(
            'ai-text-topic', 
            key=str(hash(prompt)),
            value=json.dumps(message_data),
            callback=delivery_report
        )
        producer.poll(0)
        time.sleep(3)  
except KeyboardInterrupt:
    print("\nProducer stopped by user")
finally:
    producer.flush()
    print("Producer connection closed")