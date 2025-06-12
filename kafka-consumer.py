from confluent_kafka import Consumer, KafkaError
import json
import torch
import torch.nn as nn
import numpy as np
import re
from datetime import datetime
class TransformerTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, hidden_dim=512, num_layers=4, seq_length=50):
        super(TransformerTextGenerator, self).__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(seq_length, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        batch_size, seq_len = x.shape
        embedded = self.embedding(x) + self.pos_encoding[:seq_len]
        embedded = self.dropout(embedded)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        mask = mask.to(x.device)
        output = self.transformer(embedded, mask=mask)
        output = self.output_layer(output)
        return output
class AITextGenerator:
    def __init__(self, model_path='generative_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on device: {self.device}")
        model_data = torch.load(model_path, map_location=self.device)
        self.vocab = model_data['vocab']
        self.vocab_size = model_data['vocab_size']
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        config = model_data['model_config']
        self.model = TransformerTextGenerator(
            vocab_size=self.vocab_size,
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            seq_length=config['seq_length']
        ).to(self.device)
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.eval()
        print(f"Model loaded successfully! Vocabulary size: {self.vocab_size}")
    def preprocess_text(self, text):
        """Preprocess input text"""
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return text.split()
    def text_to_sequence(self, text, max_length=50):
        words = self.preprocess_text(text)
        sequence = [self.word_to_idx.get(word, 0) for word in words]
        if len(sequence) < max_length:
            sequence = sequence + [0] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        return torch.tensor([sequence], dtype=torch.long).to(self.device)
    def generate_text(self, prompt, max_length=20, temperature=0.8, top_k=50):
        with torch.no_grad():
            input_sequence = self.text_to_sequence(prompt)
            generated_words = self.preprocess_text(prompt)
            for _ in range(max_length):
                outputs = self.model(input_sequence)
                next_token_logits = outputs[0, -1, :] / temperature
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                probabilities = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1).item()
                next_word = self.idx_to_word.get(next_token, '<UNK>')
                if next_word == '<UNK>':
                    break
                generated_words.append(next_word)
                new_sequence = torch.cat([
                    input_sequence[:, 1:], 
                    torch.tensor([[next_token]], device=self.device)
                ], dim=1)
                input_sequence = new_sequence
            return ' '.join(generated_words)
try:
    text_generator = AITextGenerator()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run train_model.py first to create the model.")
    exit(1)
config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'ai-text-consumer-group',
    'auto.offset.reset': 'earliest',
    'client.id': 'ai-text-consumer'
}
consumer = Consumer(config)
consumer.subscribe(['ai-text-topic'])
print("AI Text Generator Consumer (Confluent) started, waiting for prompts...")
print("=" * 60)
try:
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                print(f'End of partition reached {msg.topic()}/{msg.partition()}')
            else:
                print(f'Error: {msg.error()}')
                continue
        try:
            data = json.loads(msg.value().decode('utf-8'))
            prompt = data['prompt']
            max_length = data['max_length']
            temperature = data['temperature']
            top_k = data['top_k']
            timestamp = data['timestamp']
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing prompt...")
            print(f"Prompt: '{prompt}'")
            print(f"Settings: max_length={max_length}, temperature={temperature}, top_k={top_k}")
            generated_text = text_generator.generate_text(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k
            )
            print(f"Generated: '{generated_text}'")
            print("-" * 60)
        except Exception as e:
            print(f"Error processing message: {e}")
except KeyboardInterrupt:
    print("\nConsumer stopped by user")
finally:
    consumer.close()
    print("Consumer connection closed")