"""
Inference script for BERT-CNN-Attention model.

Usage:
    python infer_bert_cnn_attention.py
    python infer_bert_cnn_attention.py --dev_file test_rehydrated.jsonl --zip_name submission_cnn_attn.zip
"""

import json
import torch
import zipfile
import argparse
from tqdm import tqdm
from transformers import BertTokenizerFast
from collections import Counter

# Import model class from training script
from train_bert_cnn import BertCNNAttention


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
    parser = argparse.ArgumentParser(description='Inference for BERT-CNN-Attention model')
    parser.add_argument('--model_path', default='bert-cnn-attention-model/best_model.pt', 
                        help='Path to trained model')
    parser.add_argument('--dev_file', default='dev_rehydrated.jsonl', 
                        help='Dev/test data file')
    parser.add_argument('--output_file', default='submission.jsonl', 
                        help='Output submission file')
    parser.add_argument('--zip_name', default='submission.zip', 
                        help='Output zip file name')
    args = parser.parse_args()
    
    # Configuration (must match training)
    CONFIG = {
        'bert_model': 'bert-base-uncased',
        'max_length': 256,
        'num_filters': 128,
        'filter_sizes': [2, 3, 4, 5],
        'num_attention_heads': 4,
        'dropout': 0.4,
        'model_path': args.model_path,
        'dev_file': args.dev_file,
        'output_file': args.output_file
    }
    
    id_to_label = {0: 'No', 1: 'Yes'}
    
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
    print(f"Loading model from {CONFIG['model_path']}...")
    model = BertCNNAttention(
        bert_model_name=CONFIG['bert_model'],
        num_filters=CONFIG['num_filters'],
        filter_sizes=CONFIG['filter_sizes'],
        num_attention_heads=CONFIG['num_attention_heads'],
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
            # Tokenize
            encoding = tokenizer(
                item['text'],
                truncation=True,
                padding='max_length',
                max_length=CONFIG['max_length'],
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Predict
            logits, _ = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1).item()
            
            predictions.append({
                '_id': item['_id'],
                'conspiracy': id_to_label[pred],
                'markers': []
            })
    
    # Save submission
    print(f"Saving predictions to {CONFIG['output_file']}...")
    with open(CONFIG['output_file'], 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    # Create zip (always name file 'submission.jsonl' inside zip for CodaBench)
    zip_file = args.zip_name
    with zipfile.ZipFile(zip_file, 'w') as zf:
        zf.write(CONFIG['output_file'], 'submission.jsonl')
    print(f"Created {zip_file}")
    
    # Print distribution
    dist = Counter(p['conspiracy'] for p in predictions)
    print(f"\nPrediction distribution: {dict(dist)}")
    print(f"\nDone! Submit {zip_file} to CodaBench.")


if __name__ == '__main__':
    main()
