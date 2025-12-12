"""
Inference script for BERT-BiLSTM-MultiHeadAttention model.

Usage:
    python infer_hybrid.py
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
import zipfile

# Import model class from training script
from train_hybrid import BertBiLSTMAttention, ConspiracyDataset


def load_test_data(file_path):
    """Load test/dev data for inference."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Add dummy label if not present (for test set)
            if 'conspiracy' not in item:
                item['conspiracy'] = 'No'  # Dummy label
            data.append(item)
    return data


def inference(model, dataloader, device):
    """Run inference and return predictions."""
    model.eval()
    all_preds = []
    id_to_label = {0: 'No', 1: 'Yes'}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend([id_to_label[p.item()] for p in preds])
    
    return all_preds


def main():
    # Configuration
    CONFIG = {
        'bert_model': 'bert-base-uncased',
        'max_length': 256,
        'batch_size': 32,
        'lstm_hidden_size': 256,
        'lstm_layers': 2,
        'num_attention_heads': 8,
        'dropout': 0.3,
        'model_path': 'bert-lstm-attention-model/best_model.pt',
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
    
    # Dataset and DataLoader
    test_dataset = ConspiracyDataset(test_data, tokenizer, CONFIG['max_length'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Load model
    print("Loading model...")
    model = BertBiLSTMAttention(
        bert_model_name=CONFIG['bert_model'],
        lstm_hidden_size=CONFIG['lstm_hidden_size'],
        lstm_layers=CONFIG['lstm_layers'],
        num_attention_heads=CONFIG['num_attention_heads'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    print("Model loaded!")
    
    # Run inference
    print("Running inference...")
    predictions = inference(model, test_loader, device)
    
    # Create submission
    print(f"Creating submission file: {CONFIG['output_file']}")
    with open(CONFIG['output_file'], 'w') as f:
        for item, pred in zip(test_data, predictions):
            submission = {
                '_id': item['_id'],
                'conspiracy': pred,
                'markers': []  # Empty for binary classification
            }
            f.write(json.dumps(submission) + '\n')
    
    # Create zip
    zip_file = 'submission_hybrid.zip'
    with zipfile.ZipFile(zip_file, 'w') as zf:
        zf.write(CONFIG['output_file'])
    print(f"Created {zip_file}")
    
    # Print prediction distribution
    from collections import Counter
    dist = Counter(predictions)
    print(f"\nPrediction distribution: {dict(dist)}")
    print("\nDone! Submit submission_hybrid.zip to CodaBench.")


if __name__ == '__main__':
    main()
