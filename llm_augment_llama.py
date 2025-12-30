"""
LLM-based data augmentation using Llama-3.2-1B.
Designed to run on Kaggle/Colab with T4 GPU (16GB).

Usage (in Kaggle notebook):
    # First, login to HuggingFace
    from huggingface_hub import login
    login(token="your_token_here")
    
    # Then run augmentation
    !python Data_Augmentation/llm_augment_llama.py --sample 50  # Test first
    !python Data_Augmentation/llm_augment_llama.py              # Full run
"""

import json
import argparse
import random
from pathlib import Path
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def load_model(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    """Load Llama model - fits in 16GB GPU."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded on {model.device}")
    return model, tokenizer


def paraphrase_text(model, tokenizer, text, max_length=512):
    """Generate paraphrase using Llama."""
    
    # Simple, clear prompt
    prompt = f"""Rewrite this Reddit comment in different words. Keep the same meaning and tone. Only output the rewritten text.

Original: {text}

Rewritten:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract only the new part
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract text after "Rewritten:"
    if "Rewritten:" in full_output:
        result = full_output.split("Rewritten:")[-1].strip()
    else:
        result = full_output[len(prompt):].strip()
    
    # Clean up
    result = result.split("\n\n")[0].strip()
    result = result.split("Original:")[0].strip()  # Remove if model repeats
    
    # Validate
    if len(result) < 20 or len(result) > len(text) * 3:
        return None
    
    return result


def paraphrase_conspiracy_aware(model, tokenizer, text, max_length=512):
    """Generate paraphrase preserving conspiracy tone."""
    
    prompt = f"""Rewrite this Reddit comment while keeping:
- The same meaning and claims
- The suspicious or skeptical tone
- References to hidden agendas or cover-ups

Only output the rewritten text, nothing else.

Original: {text}

Rewritten:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Rewritten:" in full_output:
        result = full_output.split("Rewritten:")[-1].strip()
    else:
        result = full_output[len(prompt):].strip()
    
    result = result.split("\n\n")[0].strip()
    result = result.split("Original:")[0].strip()
    
    if len(result) < 20 or len(result) > len(text) * 3:
        return None
    
    return result


def load_data(file_path, exclude_labels=None):
    """Load JSONL dataset."""
    exclude_labels = exclude_labels or []
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if item.get('conspiracy') not in exclude_labels and item.get('conspiracy') is not None:
                data.append(item)
    return data


def save_data(data, file_path):
    """Save dataset to JSONL."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Llama-based augmentation')
    parser.add_argument('--input', default='train_rehydrated.jsonl', help='Input file')
    parser.add_argument('--output', default='data/augmented/train_llama_augmented.jsonl', help='Output file')
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B-Instruct', help='Model name')
    parser.add_argument('--target', nargs='+', default=['Yes'], help='Labels to augment (default: Yes only)')
    parser.add_argument('--num_aug', type=int, default=1, help='Augmentations per sample')
    parser.add_argument('--sample', type=int, default=None, help='Sample N items for testing')
    parser.add_argument('--conspiracy_aware', action='store_true', help='Use conspiracy-aware prompt')
    parser.add_argument('--exclude', nargs='+', default=["Can't tell"], help='Labels to exclude')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load data
    print(f"Loading data from {args.input}")
    data = load_data(args.input, exclude_labels=args.exclude)
    print(f"Loaded {len(data)} samples")
    print(f"Distribution: {Counter(d['conspiracy'] for d in data)}")
    
    # Filter to target labels
    if args.target:
        candidates = [d for d in data if d['conspiracy'] in args.target]
    else:
        candidates = data.copy()
    
    # Sample if specified
    if args.sample and args.sample < len(candidates):
        candidates = random.sample(candidates, args.sample)
    
    print(f"\nAugmenting {len(candidates)} samples ({args.target})")
    print(f"Augmentations per sample: {args.num_aug}")
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Choose paraphrase function
    paraphrase_fn = paraphrase_conspiracy_aware if args.conspiracy_aware else paraphrase_text
    print(f"Using {'conspiracy-aware' if args.conspiracy_aware else 'standard'} paraphrasing")
    
    # Augment
    augmented_data = data.copy()  # Start with all original data
    successful = 0
    failed = 0
    
    for item in tqdm(candidates, desc="Augmenting"):
        for aug_idx in range(args.num_aug):
            try:
                aug_text = paraphrase_fn(model, tokenizer, item['text'])
                
                if aug_text:
                    aug_item = {
                        '_id': f"{item['_id']}_llama{aug_idx}",
                        'text': aug_text,
                        'subreddit': item.get('subreddit', ''),
                        'conspiracy': item['conspiracy'],
                        'markers': [],
                        'annotator': item.get('annotator', ''),
                        'augmentation': 'llama_paraphrase'
                    }
                    augmented_data.append(aug_item)
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error: {e}")
                failed += 1
    
    print(f"\n{'='*50}")
    print(f"Augmentation complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Original samples: {len(data)}")
    print(f"Total samples: {len(augmented_data)}")
    print(f"Distribution: {Counter(d['conspiracy'] for d in augmented_data)}")
    
    # Save
    save_data(augmented_data, args.output)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
