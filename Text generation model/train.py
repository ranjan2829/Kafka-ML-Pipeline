import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import joblib
from collections import Counter
import re

class TextDataset(Dataset):
    def __init__(self, texts, vocab, seq_length=50):
        self.texts = texts
        self.vocab = vocab
        self.seq_length = seq_length
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
    def __len__(self):
        return len(self.texts) - self.seq_length
    
    def __getitem__(self, idx):
        sequence = self.texts[idx:idx + self.seq_length]
        target = self.texts[idx + 1:idx + self.seq_length + 1]
        
        sequence_indices = [self.word_to_idx.get(word, 0) for word in sequence]
        target_indices = [self.word_to_idx.get(word, 0) for word in target]
        
        return torch.tensor(sequence_indices), torch.tensor(target_indices)

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
        
        # Add embeddings and positional encoding
        embedded = self.embedding(x) + self.pos_encoding[:seq_len]
        embedded = self.dropout(embedded)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        mask = mask.to(x.device)
        
        # Apply transformer
        output = self.transformer(embedded, mask=mask)
        output = self.output_layer(output)
        
        return output

def preprocess_text(text):
    """Simple text preprocessing"""
    # Convert to lowercase and split into words
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    words = text.split()
    return words

def create_sample_dataset():
    """Create a sample dataset of creative texts"""
    sample_texts = [
        "The mystical forest whispered ancient secrets to those who dared to listen carefully",
        "In the digital realm, algorithms dance with data creating beautiful patterns of logic",
        "Time flows like a river carrying memories from past to future generations",
        "Artificial intelligence learns to understand human creativity and expression through practice",
        "The quantum world reveals mysteries that challenge our understanding of reality itself",
        "Deep learning networks process information in ways similar to human brain functions",
        "Stories have the power to transport minds across vast distances and time periods",
        "Technology bridges gaps between imagination and reality through innovative solutions",
        "Neural networks discover hidden patterns in data that humans might never notice",
        "The future holds endless possibilities for those brave enough to explore them",
        "Machine learning algorithms evolve and adapt to solve complex real world problems",
        "Creativity emerges from the intersection of knowledge experience and imagination combined",
        "Data scientists work with vast amounts of information to extract meaningful insights",
        "Innovation happens when diverse minds collaborate on challenging problems together",
        "The digital age transforms how we communicate learn and interact with each other",
        "Generative models create new content by learning from existing patterns and structures",
        "Transformers revolutionized natural language processing with attention mechanisms that focus",
        "Deep neural networks can generate realistic images text and audio from learned representations"
    ]
    
    all_words = []
    for text in sample_texts:
        all_words.extend(preprocess_text(text))
    
    return all_words

def train_model():
    print("Creating sample dataset...")
    texts = create_sample_dataset()
    
    print("Building vocabulary...")
    word_counts = Counter(texts)
    vocab = ['<UNK>'] + [word for word, count in word_counts.most_common(1000)]
    vocab_size = len(vocab)
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataset
    dataset = TextDataset(texts, vocab)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerTextGenerator(vocab_size).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on device: {device}")
    print("Starting training...")
    
    model.train()
    for epoch in range(100):  # More epochs for better learning
        total_loss = 0
        for batch_idx, (sequences, targets) in enumerate(dataloader):
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    print("Training completed!")
    
    # Save model and vocabulary
    model_data = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'vocab_size': vocab_size,
        'model_config': {
            'embed_dim': 128,
            'num_heads': 8,
            'hidden_dim': 512,
            'num_layers': 4,
            'seq_length': 50
        }
    }
    
    torch.save(model_data, 'generative_model.pth')
    print("Model saved as 'generative_model.pth'")

if __name__ == "__main__":
    train_model()