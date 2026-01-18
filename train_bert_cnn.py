"""
Hybrid BERT-CNN-Attention model for conspiracy detection.

Architecture:
    Input -> BERT -> Multi-scale CNN -> Attention -> Classification

The CNN filters capture n-gram patterns, and attention weights
the importance of each filter's output.

Usage:
    python train_bert_cnn_attention.py
    python train_bert_cnn_attention.py --train_file data/augmented/train_llama3b_conspiracy_fixed.jsonl
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
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
        
        conspiracy_label = item.get('conspiracy')
        if conspiracy_label in self.label_map:
            label = self.label_map[conspiracy_label]
        else:
            label = 0
        
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


class CNNAttention(nn.Module):
    """
    Attention mechanism over CNN filter outputs.
    
    Learns to weight the importance of different n-gram patterns.
    """
    
    def __init__(self, num_filters, num_filter_sizes):
        super().__init__()
        total_filters = num_filters * num_filter_sizes
        
        # Attention layers
        self.attention = nn.Sequential(
            nn.Linear(total_filters, total_filters // 2),
            nn.Tanh(),
            nn.Linear(total_filters // 2, 1)
        )
    
    def forward(self, conv_outputs):
        """
        Args:
            conv_outputs: List of tensors, each of shape (batch, num_filters, seq_len_i)
        
        Returns:
            attended: (batch, total_filters) - attention-weighted representation
            attention_weights: (batch, num_positions) - attention weights for visualization
        """
        batch_size = conv_outputs[0].size(0)
        
        # For each filter size, we have (batch, num_filters, variable_seq_len)
        # We'll attend over all positions across all filter sizes
        
        all_features = []
        all_positions = []
        
        for conv_out in conv_outputs:
            # conv_out: (batch, num_filters, seq_len)
            # Transpose to (batch, seq_len, num_filters)
            features = conv_out.transpose(1, 2)
            all_features.append(features)
            all_positions.append(features.size(1))
        
        # Pad and concatenate along sequence dimension
        max_len = max(all_positions)
        padded_features = []
        
        for features in all_features:
            if features.size(1) < max_len:
                padding = torch.zeros(
                    batch_size, max_len - features.size(1), features.size(2),
                    device=features.device
                )
                features = torch.cat([features, padding], dim=1)
            padded_features.append(features)
        
        # Stack: (batch, num_filter_sizes, max_len, num_filters)
        stacked = torch.stack(padded_features, dim=1)
        
        # Reshape: (batch, num_filter_sizes * max_len, num_filters)
        batch_size, num_sizes, seq_len, num_filters = stacked.size()
        flat_features = stacked.view(batch_size, num_sizes * seq_len, num_filters)
        
        # Compute attention scores
        attn_scores = self.attention(flat_features).squeeze(-1)  # (batch, num_positions)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, num_positions)
        
        # Apply attention
        attended = torch.bmm(attn_weights.unsqueeze(1), flat_features).squeeze(1)  # (batch, num_filters)
        
        return attended, attn_weights


class SelfAttentionOverFilters(nn.Module):
    """
    Multi-head self-attention over CNN filter outputs.
    
    Treats each filter size's max-pooled output as a "token" and 
    applies self-attention to model interactions between different n-gram scales.
    """
    
    def __init__(self, num_filters, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = num_filters // num_heads
        
        assert num_filters % num_heads == 0, "num_filters must be divisible by num_heads"
        
        self.query = nn.Linear(num_filters, num_filters)
        self.key = nn.Linear(num_filters, num_filters)
        self.value = nn.Linear(num_filters, num_filters)
        self.out = nn.Linear(num_filters, num_filters)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_filter_sizes, num_filters) - max-pooled outputs from each CNN filter size
        
        Returns:
            output: (batch, num_filter_sizes, num_filters) - attention-refined representations
            attn_weights: (batch, num_heads, num_filter_sizes, num_filter_sizes)
        """
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
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        output = self.out(context)
        
        return output, attn_weights


class BertCNNAttention(nn.Module):
    """
    BERT + Multi-scale CNN + Attention model.
    
    Architecture:
        1. BERT encodes the input text
        2. Multiple CNN filters (sizes 2,3,4,5) capture n-gram patterns
        3. Self-attention models interactions between different n-gram scales
        4. Classifier predicts conspiracy label
    
    The attention helps the model learn which n-gram patterns are most
    important for conspiracy detection.
    """
    
    def __init__(
        self,
        bert_model_name='bert-base-uncased',
        num_filters=128,
        filter_sizes=[2, 3, 4, 5],
        num_attention_heads=4,
        num_classes=2,
        dropout=0.4,
        freeze_bert=False
    ):
        super().__init__()
        
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        
        # BERT encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Multi-scale CNN
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=bert_hidden_size,
                out_channels=num_filters,
                kernel_size=fs,
                padding=fs // 2  # Same padding to preserve sequence length
            )
            for fs in filter_sizes
        ])
        
        # Batch normalization for each conv
        self.conv_bns = nn.ModuleList([
            nn.BatchNorm1d(num_filters)
            for _ in filter_sizes
        ])
        
        # Self-attention over filter outputs
        self.self_attention = SelfAttentionOverFilters(
            num_filters=num_filters,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(num_filters)
        
        total_filters = num_filters * len(filter_sizes)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(total_filters, total_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_filters // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        # BERT encoding
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = bert_output.last_hidden_state  # (batch, seq_len, 768)
        
        # Transpose for CNN: (batch, 768, seq_len)
        x = sequence_output.transpose(1, 2)
        
        # Apply CNN filters and batch norm
        conv_outputs = []
        for conv, bn in zip(self.convs, self.conv_bns):
            conv_out = F.relu(bn(conv(x)))  # (batch, num_filters, seq_len)
            # Max pool over sequence
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch, num_filters)
            conv_outputs.append(pooled)
        
        # Stack filter outputs: (batch, num_filter_sizes, num_filters)
        stacked = torch.stack(conv_outputs, dim=1)
        
        # Self-attention over different n-gram scales
        attended, attn_weights = self.self_attention(stacked)
        
        # Residual connection + layer norm
        attended = self.layer_norm(attended + stacked)
        
        # Flatten: (batch, num_filter_sizes * num_filters)
        flat = attended.view(attended.size(0), -1)
        
        # Classification
        flat = self.dropout(flat)
        logits = self.classifier(flat)
        
        return logits, attn_weights


def load_data(file_path, exclude_labels=None):
    """Load JSONL data."""
    exclude_labels = exclude_labels or []
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            conspiracy = item.get('conspiracy')
            
            if conspiracy is None:
                continue
            if conspiracy in exclude_labels:
                continue
                
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


def main():
    parser = argparse.ArgumentParser(description='Train BERT-CNN-Attention model')
    parser.add_argument('--train_file', default='train_rehydrated.jsonl', help='Training data file')
    parser.add_argument('--output_dir', default='bert-cnn-attention-model', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--num_filters', type=int, default=128, help='Number of CNN filters')
    parser.add_argument('--num_attention_heads', type=int, default=4, help='Number of attention heads')
    args = parser.parse_args()
    
    # Configuration
    CONFIG = {
        'bert_model': 'bert-base-uncased',
        'max_length': 256,
        'batch_size': args.batch_size,
        'num_filters': args.num_filters,
        'filter_sizes': [2, 3, 4, 5],
        'num_attention_heads': args.num_attention_heads,
        'dropout': args.dropout,
        'learning_rate': args.lr,
        'num_epochs': args.epochs,
        'warmup_ratio': 0.1,
        'freeze_bert': False,
        'train_file': args.train_file,
        'output_dir': args.output_dir,
        'exclude_labels': ["Can't tell"]
    }
    
    # Create output directory
    import os
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training file: {CONFIG['train_file']}")
    
    # Load data
    print("Loading data...")
    train_data = load_data(CONFIG['train_file'], exclude_labels=CONFIG['exclude_labels'])
    print(f"Train samples: {len(train_data)}")
    
    # Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(CONFIG['bert_model'])
    
    # Dataset
    train_dataset = ConspiracyDataset(train_data, tokenizer, CONFIG['max_length'])
    
    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # Model
    print("Initializing BERT-CNN-Attention model...")
    model = BertCNNAttention(
        bert_model_name=CONFIG['bert_model'],
        num_filters=CONFIG['num_filters'],
        filter_sizes=CONFIG['filter_sizes'],
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
    print(f"Architecture: BERT -> CNN (filters: {CONFIG['filter_sizes']}) -> Self-Attention ({CONFIG['num_attention_heads']} heads) -> Classifier")
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")
        print('='*50)
        
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"{CONFIG['output_dir']}/model_epoch_{epoch+1}.pt")
    
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print('='*50)
    
    # Save final model
    torch.save(model.state_dict(), f"{CONFIG['output_dir']}/best_model.pt")
    
    # Save model with config
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG
    }, f"{CONFIG['output_dir']}/final_model.pt")
    print(f"Model saved to {CONFIG['output_dir']}/")


if __name__ == '__main__':
    main()
