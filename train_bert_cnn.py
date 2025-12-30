"""
Hybrid BERT-CNN model for conspiracy detection (no validation split version).

Architecture:
    Input -> BERT -> Multi-scale CNN (multiple filter sizes) -> Classification

Usage:
    python train_bert_cnn_no_val.py
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class BertCNN(nn.Module):
    """
    BERT + Multi-scale CNN model.
    
    The CNN filters of different sizes capture n-gram patterns:
        - Filter size 2: bigrams
        - Filter size 3: trigrams
        - Filter size 4: 4-grams
        - Filter size 5: 5-grams
    """
    
    def __init__(
        self,
        bert_model_name='bert-base-uncased',
        num_filters=128,
        filter_sizes=[2, 3, 4, 5],
        num_classes=2,
        dropout=0.4,
        freeze_bert=False
    ):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=bert_hidden_size,
                out_channels=num_filters,
                kernel_size=fs
            )
            for fs in filter_sizes
        ])
        
        total_filters = num_filters * len(filter_sizes)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(total_filters, total_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_filters // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = bert_output.last_hidden_state
        
        x = sequence_output.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        cat = torch.cat(conv_outputs, dim=1)
        cat = self.dropout(cat)
        logits = self.classifier(cat)
        
        return logits


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
        
        logits = model(input_ids, attention_mask)
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
    # Configuration
    CONFIG = {
        'bert_model': 'bert-base-uncased',
        'max_length': 256,
        'batch_size': 16,
        'num_filters': 128,
        'filter_sizes': [2, 3, 4, 5],
        'dropout': 0.4,
        'learning_rate': 2e-5,
        'num_epochs': 10,  # Fixed epochs like BERT-LSTM run
        'warmup_ratio': 0.1,
        'freeze_bert': False,
        'train_file': 'train_rehydrated.jsonl',
        'output_dir': 'bert-cnn-model',
        'exclude_labels': ["Can't tell"]
    }
    
    # Create output directory
    import os
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data (full dataset, no split)
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
    print("Initializing BERT-CNN model...")
    model = BertCNN(
        bert_model_name=CONFIG['bert_model'],
        num_filters=CONFIG['num_filters'],
        filter_sizes=CONFIG['filter_sizes'],
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
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")
        print('='*50)
        
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
    
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
