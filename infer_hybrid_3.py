"""
Inference script for BERT-BiLSTM-MultiHeadAttention model (3-class).

Usage:
    python infer_hybrid_3class.py
"""

import json
import torch
import zipfile
from tqdm import tqdm
from transformers import BertTokenizerFast
from collections import Counter

from train_hybrid_3class import BertBiLSTMAttention, NUM_CLASSES

# 3-class mapping
ID_TO_LABEL = {0: "No", 1: "Yes", 2: "Can't tell"}


def load_test_data(file_path):
    """Load test/dev data for inference (no labels needed)."""
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                sample_id = item.get("_id", f"sample_{i}")
                data.append({
                    "_id": sample_id,
                    "text": item.get("text", "")
                })
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line at index {i}")
    return data


def main():
    # Configuration - must match training
    CONFIG = {
        'bert_model': 'bert-base-uncased',
        'max_length': 256,
        'lstm_hidden_size': 256,
        'lstm_layers': 2,
        'num_attention_heads': 8,
        'num_classes': NUM_CLASSES,  # 3 classes
        'dropout': 0.3,
        'model_path': 'bert-lstm-attention-3class/best_model.pt',
        'dev_file': 'dev_rehydrated.jsonl',
        'output_file': 'submission.jsonl'
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print(f"Loading data from {CONFIG['dev_file']}...")
    test_data = load_test_data(CONFIG['dev_file'])
    print(f"Loaded {len(test_data)} samples")
    
    # Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(CONFIG['bert_model'])
    
    # Load model
    print("Loading model...")
    model = BertBiLSTMAttention(
        bert_model_name=CONFIG['bert_model'],
        lstm_hidden_size=CONFIG['lstm_hidden_size'],
        lstm_layers=CONFIG['lstm_layers'],
        num_attention_heads=CONFIG['num_attention_heads'],
        num_classes=CONFIG['num_classes'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    model.eval()
    print("Model loaded!")
    
    # Run inference
    print("Running inference...")
    predictions = []
    
    with torch.no_grad():
        for item in tqdm(test_data, desc="Inference"):
            encoding = tokenizer(
                item['text'],
                truncation=True,
                padding='max_length',
                max_length=CONFIG['max_length'],
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1).item()
            
            predictions.append({
                '_id': item['_id'],
                'conspiracy': ID_TO_LABEL[pred],
                'markers': []
            })
    
    # Save submission
    print(f"Saving predictions to {CONFIG['output_file']}...")
    with open(CONFIG['output_file'], 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    # Create zip
    zip_file = 'submission_3class.zip'
    with zipfile.ZipFile(zip_file, 'w') as zf:
        zf.write(CONFIG['output_file'], 'submission.jsonl')
    print(f"Created {zip_file}")
    
    # Print distribution
    dist = Counter(p['conspiracy'] for p in predictions)
    print(f"\nPrediction distribution: {dict(dist)}")
    print("\nDone! Submit submission_3class.zip to CodaBench.")


if __name__ == '__main__':
    main()
