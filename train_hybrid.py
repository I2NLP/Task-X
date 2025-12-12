"""
Hybrid BERT-BiLSTM-MultiHeadAttention model for conspiracy detection.

Architecture:
    Input -> BERT -> BiLSTM -> Multi-Head Attention -> Classification

Usage:
    python train_hybrid.py
"""

import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm


class ConspiracyDataset(Dataset):
    """Dataset for conspiracy detection."""
    
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"No": 0, "Yes": 1}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = self.label_map[item['conspiracy']]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer."""
    
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
    
    def forward(self, x, mask=None):
        batch_size, seq_len, hidden_size = x.size()
        
        # Linear projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Output projection
        output = self.out(context)
        
        return output, attn_weights


class BertBiLSTMAttention(nn.Module):
    """
    Hybrid model: BERT + BiLSTM + Multi-Head Attention
    
    Architecture:
        BERT embeddings -> BiLSTM -> Multi-Head Attention -> Pooling -> Classifier
    """
    
    def __init__(
        self,
        bert_model_name='bert-base-uncased',
        lstm_hidden_size=256,
        lstm_layers=2,
        num_attention_heads=8,
        num_classes=2,
        dropout=0.3,
        freeze_bert=False
    ):
        super().__init__()
        
        # BERT encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        
        # Optionally freeze BERT
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        lstm_output_size = lstm_hidden_size * 2  # Bidirectional
        
        # Multi-Head Attention
        self.attention = MultiHeadAttention(
            hidden_size=lstm_output_size,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        # BERT encoding
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = bert_output.last_hidden_state  # (batch, seq_len, 768)
        
        # BiLSTM
        lstm_output, _ = self.lstm(sequence_output)  # (batch, seq_len, lstm_hidden*2)
        
        # Multi-Head Attention
        attn_output, attn_weights = self.attention(lstm_output, mask=attention_mask)
        
        # Residual connection + layer norm
        attn_output = self.layer_norm(attn_output + lstm_output)
        
        # Pooling: use attention-weighted mean
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_output = torch.sum(attn_output * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_output / sum_mask
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits, attn_weights


def load_data(file_path, exclude_labels=None):
    """Load JSONL data."""
    exclude_labels = exclude_labels or []
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if item.get('conspiracy') not in exclude_labels:
                data.append(item)
    return data


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, f1


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, f1, all_preds, all_labels


def main():
    # Configuration
    CONFIG = {
        'bert_model': 'bert-base-uncased',
        'max_length': 256,
        'batch_size': 16,
        'lstm_hidden_size': 256,
        'lstm_layers': 2,
        'num_attention_heads': 8,
        'dropout': 0.3,
        'learning_rate': 2e-5,
        'num_epochs': 5,
        'warmup_ratio': 0.1,
        'freeze_bert': False,
        'train_file': 'train_rehydrated.jsonl',
        'dev_file': 'dev_rehydrated.jsonl',
        'output_dir': 'bert-lstm-attention-model',
        'exclude_labels': ["Can't tell"]
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_data = load_data(CONFIG['train_file'], exclude_labels=CONFIG['exclude_labels'])
    dev_data = load_data(CONFIG['dev_file'], exclude_labels=CONFIG['exclude_labels'])
    print(f"Train samples: {len(train_data)}, Dev samples: {len(dev_data)}")
    
    # Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(CONFIG['bert_model'])
    
    # Datasets
    train_dataset = ConspiracyDataset(train_data, tokenizer, CONFIG['max_length'])
    dev_dataset = ConspiracyDataset(dev_data, tokenizer, CONFIG['max_length'])
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Model
    print("Initializing model...")
    model = BertBiLSTMAttention(
        bert_model_name=CONFIG['bert_model'],
        lstm_hidden_size=CONFIG['lstm_hidden_size'],
        lstm_layers=CONFIG['lstm_layers'],
        num_attention_heads=CONFIG['num_attention_heads'],
        dropout=CONFIG['dropout'],
        freeze_bert=CONFIG['freeze_bert']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with class weights
    train_labels = [item['conspiracy'] for item in train_data]
    class_counts = {label: train_labels.count(label) for label in ['No', 'Yes']}
    total = len(train_labels)
    class_weights = torch.tensor([total / class_counts['No'], total / class_counts['Yes']], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: {class_weights}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    
    # Scheduler
    total_steps = len(train_loader) * CONFIG['num_epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    # Training loop
    print("\nStarting training...")
    best_f1 = 0
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")
        print('='*50)
        
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        
        dev_loss, dev_f1, dev_preds, dev_labels = evaluate(model, dev_loader, criterion, device)
        print(f"Dev Loss: {dev_loss:.4f}, Dev F1: {dev_f1:.4f}")
        
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model.state_dict(), f"{CONFIG['output_dir']}/best_model.pt")
            print(f"New best model saved! F1: {best_f1:.4f}")
    
    print(f"\n{'='*50}")
    print(f"Training complete! Best Dev F1: {best_f1:.4f}")
    print('='*50)
    
    # Final evaluation with classification report
    print("\nFinal Classification Report:")
    print(classification_report(dev_labels, dev_preds, target_names=['No', 'Yes']))
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG
    }, f"{CONFIG['output_dir']}/final_model.pt")
    print(f"Model saved to {CONFIG['output_dir']}/")


if __name__ == '__main__':
    import os
    os.makedirs('bert-lstm-attention-model', exist_ok=True)
    main()
