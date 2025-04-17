import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
import numpy as np
from collections import defaultdict, Counter
import random

## 1. Dataset Preparation

class ProcessHierarchyDataset(Dataset):
    def __init__(self, input_sequences, output_sequences, input_tokenizer, output_tokenizer):
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        
    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self, idx):
        input_seq = self.input_sequences[idx]
        output_seq = self.output_sequences[idx]
        
        # Convert to token IDs
        input_ids = self.input_tokenizer.encode(input_seq)
        output_ids = self.output_tokenizer.encode(output_seq)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long),
            'input_text': input_seq,
            'output_text': output_seq
        }

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    output_ids = [item['output_ids'] for item in batch]
    
    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    output_ids = pad_sequence(output_ids, batch_first=True, padding_value=0)
    
    # Create attention masks
    input_mask = (input_ids != 0).int()
    output_mask = (output_ids != 0).int()
    
    return {
        'input_ids': input_ids,
        'output_ids': output_ids,
        'input_mask': input_mask,
        'output_mask': output_mask,
        'input_texts': [item['input_text'] for item in batch],
        'output_texts': [item['output_text'] for item in batch]
    }

## 2. Tokenizers

class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        
        # Initialize with special tokens
        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
        
        self.vocab_size = len(self.special_tokens)
    
    def build_vocab(self, sequences):
        # Count all tokens
        token_counts = Counter()
        for seq in sequences:
            tokens = self.tokenize(seq)
            token_counts.update(tokens)
        
        # Add to vocab (excluding special tokens)
        for token, _ in token_counts.most_common():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.reverse_vocab[len(self.reverse_vocab)] = token
        
        self.vocab_size = len(self.vocab)
    
    def tokenize(self, text):
        # Simple whitespace tokenizer for this example
        return text.split()
    
    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
    
    def decode(self, ids):
        return ' '.join([self.reverse_vocab.get(id, '<unk>') for id in ids])

## 3. Transformer Model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024):
        super().__init__()
        self.d_model = d_model
        
        # Input embedding and positional encoding
        self.input_embedding = nn.Embedding(input_vocab_size, d_model)
        self.input_pos_encoder = PositionalEncoding(d_model)
        
        # Output embedding and positional encoding
        self.output_embedding = nn.Embedding(output_vocab_size, d_model)
        self.output_pos_encoder = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward
        )
        
        # Final projection
        self.fc_out = nn.Linear(d_model, output_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # Embed and add positional encoding
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        src = self.input_pos_encoder(src)
        
        tgt = self.output_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.output_pos_encoder(tgt)
        
        # Transformer expects (seq_len, batch, d_model)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        
        # Generate square subsequent mask for tgt
        seq_len = tgt.size(0)
        tgt_mask = self.transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        
        # Forward pass through transformer
        output = self.transformer(
            src, tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project back to vocabulary size
        output = self.fc_out(output)
        
        return output

## 4. Training Utilities

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        src = batch['input_ids'].to(device)
        tgt = batch['output_ids'].to(device)
        src_mask = batch['input_mask'].to(device)
        tgt_mask = batch['output_mask'].to(device)
        
        # Shift tgt for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt_input, src_key_padding_mask=~src_mask.bool(), tgt_key_padding_mask=~tgt_mask[:, :-1].bool())
        
        # Calculate loss
        loss = criterion(output.transpose(1, 2), tgt_output)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['input_ids'].to(device)
            tgt = batch['output_ids'].to(device)
            src_mask = batch['input_mask'].to(device)
            tgt_mask = batch['output_mask'].to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = model(src, tgt_input, src_key_padding_mask=~src_mask.bool(), tgt_key_padding_mask=~tgt_mask[:, :-1].bool())
            
            loss = criterion(output.transpose(1, 2), tgt_output)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

## 5. Main Training Loop

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sample data (in practice, you'd load your actual dataset)
    input_sequences = [
        "1 1 1 0",
        "1 1 1 0 2 1 1 1",
        "1 1 1 0 2 1 1 1 3 3 3 2",
        # Add more samples...
    ]
    
    output_sequences = [
        "0: 0 : init() : pid = 1",
        "0: 0 : init() : pid = 1 1: 1 : fork() : retcode = 2",
        "0: 0 : init() : pid = 1 1: 1 : fork() : retcode = 2 2: 2 : fork() : retcode = 3 3: 3 : setsid() : retcode = 0",
        # Add more samples...
    ]
    
    # Create tokenizers
    input_tokenizer = Tokenizer()
    output_tokenizer = Tokenizer()
    
    # Build vocabularies
    input_tokenizer.build_vocab(input_sequences)
    output_tokenizer.build_vocab(output_sequences)
    
    # Create dataset and dataloader
    dataset = ProcessHierarchyDataset(input_sequences, output_sequences, input_tokenizer, output_tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # Initialize model
    model = TransformerModel(
        input_vocab_size=input_tokenizer.vocab_size,
        output_vocab_size=output_tokenizer.vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        val_loss = evaluate(model, dataloader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Save model
    torch.save(model.state_dict(), 'process_hierarchy_transformer.pth')

if __name__ == '__main__':
    main()
