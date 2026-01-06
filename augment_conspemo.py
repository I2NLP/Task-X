"""
Data augmentation using proper chat templates for instruction-tuned models.

FIXED VERSION:
- Uses proper chat templates for each model
- Correctly extracts the first paraphrase
- Handles multiple alternatives in output

Usage:
    python Data_Augmentation/augment_fixed.py --sample 20 --show --model llama3b
    python Data_Augmentation/augment_fixed.py --target Yes No --model llama3b
"""

import json
import argparse
import random
import re
from pathlib import Path
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ============================================
# MODEL CONFIGURATIONS
# ============================================
MODEL_CONFIGS = {
    'llama1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'llama3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
    'bloomz': 'bigscience/bloomz-3b',
    'qwen': 'Qwen/Qwen2.5-3B-Instruct',
    'qwen7b': 'Qwen/Qwen2.5-7B-Instruct',
}


# ============================================
# PROMPT TEMPLATES
# ============================================
CONSPIRACY_PROMPT = """Please rewrite this conspiracy-related content using different words. Keep the exact same meaning, claims, suspicious tone, and any references to cover-ups or hidden agendas. Do not soften or neutralize the content. Do not add alternatives or explanations - just provide ONE rewritten version.

Content: {text}

Rewritten version:"""

PARAPHRASE_PROMPT = """Rewrite the following text using different words while keeping the exact same meaning and tone. Provide only ONE rewritten version, no alternatives or explanations.

Text: {text}

Rewritten:"""


def load_model(model_key):
    """Load model and tokenizer."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
    
    model_name = MODEL_CONFIGS[model_key]
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


def generate_with_chat_template(model, tokenizer, text, prompt_type='conspiracy', temperature=0.7):
    """Generate using proper chat template."""
    
    # Select prompt
    if prompt_type == 'conspiracy':
        user_content = CONSPIRACY_PROMPT.format(text=text)
    else:
        user_content = PARAPHRASE_PROMPT.format(text=text)
    
    # Create messages for chat template
    messages = [{"role": "user", "content": user_content}]
    
    # Apply chat template (handles model-specific formatting)
    try:
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception:
        # Fallback for models without chat template (like BLOOMZ)
        prompt = user_content
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Calculate max tokens based on input length
    input_len = len(text.split())
    max_new_tokens = min(int(input_len * 2) + 100, 500)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    # Decode full output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return full_output, prompt


def extract_paraphrase(full_output, prompt, original_text):
    """Extract the paraphrased text from model output."""
    
    # Method 1: Remove the prompt from output
    if prompt in full_output:
        result = full_output[len(prompt):].strip()
    else:
        # Try to find where the response starts
        result = full_output
    
    # Method 2: Look for common markers and extract after them
    markers = [
        "Rewritten version:", "Rewritten:", "assistant", 
        "Here is the rewritten", "Here's the rewritten"
    ]
    
    for marker in markers:
        if marker in result:
            parts = result.split(marker)
            if len(parts) > 1:
                result = parts[-1].strip()
                break
    
    # Method 3: Remove the original text if it appears at the start
    if result.startswith(original_text[:50]):
        result = result[len(original_text):].strip()
    
    # Clean up: Take only the FIRST paraphrase (before "Or," or "Alternatively")
    stop_patterns = [
        "\n\nOr,", "\nOr,", "Or, alternatively", "Alternatively,",
        "\n\nOr ", "\nOr ", "\n\nNote:", "\nNote:",
        "\n\nHere's another", "\nHere's another",
        "\n\n---", "\n---"
    ]
    
    for pattern in stop_patterns:
        if pattern in result:
            result = result.split(pattern)[0].strip()
    
    # Remove leading/trailing quotes
    result = result.strip('"\'')
    
    # Remove any remaining prompt fragments
    prompt_fragments = [
        "Please rewrite", "Keep the exact same", "Do not soften",
        "Content:", "Text:", "Rewritten:"
    ]
    for frag in prompt_fragments:
        if result.startswith(frag):
            return None
    
    # Final cleanup - take first paragraph if multiple
    if "\n\n" in result:
        result = result.split("\n\n")[0].strip()
    
    return result if result else None


def validate_output(original, rewritten):
    """Validate the rewritten output."""
    if not rewritten:
        return False, "Empty output"
    
    # Check length
    orig_words = len(original.split())
    rewrite_words = len(rewritten.split())
    
    if rewrite_words < 5:
        return False, f"Too short ({rewrite_words} words)"
    
    ratio = rewrite_words / orig_words if orig_words > 0 else 0
    
    if ratio < 0.3:
        return False, f"Too short (ratio: {ratio:.2f})"
    if ratio > 3.0:
        return False, f"Too long (ratio: {ratio:.2f})"
    
    # Check if it's just the original
    if rewritten.strip().lower() == original.strip().lower():
        return False, "Same as original"
    
    return True, "OK"


def load_data(file_path, exclude_labels=None):
    """Load JSONL dataset."""
    exclude_labels = exclude_labels or []
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            conspiracy = item.get('conspiracy')
            if conspiracy not in exclude_labels and conspiracy is not None:
                data.append(item)
    return data


def save_data(data, file_path):
    """Save dataset to JSONL."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Fixed augmentation with chat templates')
    parser.add_argument('--input', default='train_rehydrated.jsonl', help='Input file')
    parser.add_argument('--output', default=None, help='Output file (auto-generated if not specified)')
    parser.add_argument('--model', default='llama3b', choices=list(MODEL_CONFIGS.keys()),
                        help='Model to use')
    parser.add_argument('--prompt', default='conspiracy', choices=['conspiracy', 'paraphrase'],
                        help='Prompt type')
    parser.add_argument('--target', nargs='+', default=['Yes', 'No'], help='Labels to augment')
    parser.add_argument('--num_aug', type=int, default=1, help='Augmentations per sample')
    parser.add_argument('--sample', type=int, default=None, help='Sample N items for testing')
    parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
    parser.add_argument('--exclude', nargs='+', default=["Can't tell"], help='Labels to exclude')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--show', action='store_true', help='Show examples during augmentation')
    parser.add_argument('--debug', action='store_true', help='Show raw model outputs for debugging')
    
    args = parser.parse_args()
    
    # Set output file if not specified
    if args.output is None:
        args.output = f'data/augmented/train_{args.model}_{args.prompt}_fixed.jsonl'
    
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
    
    print(f"\nAugmenting {len(candidates)} samples")
    print(f"Model: {args.model} ({MODEL_CONFIGS[args.model]})")
    print(f"Prompt type: {args.prompt}")
    print(f"Temperature: {args.temperature}")
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Augment
    augmented_data = data.copy()
    successful = 0
    failed = 0
    fail_reasons = Counter()
    
    for item in tqdm(candidates, desc="Augmenting"):
        for aug_idx in range(args.num_aug):
            try:
                # Generate
                full_output, prompt = generate_with_chat_template(
                    model, tokenizer, item['text'],
                    prompt_type=args.prompt,
                    temperature=args.temperature
                )
                
                # Debug mode: show raw output
                if args.debug:
                    print(f"\n{'='*60}")
                    print(f"ORIGINAL: {item['text'][:200]}...")
                    print(f"\nRAW OUTPUT: {full_output[-500:]}...")
                
                # Extract paraphrase
                rewritten = extract_paraphrase(full_output, prompt, item['text'])
                
                # Validate
                is_valid, reason = validate_output(item['text'], rewritten)
                
                if is_valid:
                    aug_item = {
                        '_id': f"{item['_id']}_{args.model}_{args.prompt}{aug_idx}",
                        'text': rewritten,
                        'subreddit': item.get('subreddit', ''),
                        'conspiracy': item['conspiracy'],
                        'markers': [],
                        'annotator': item.get('annotator', ''),
                        'augmentation': f'{args.model}_{args.prompt}'
                    }
                    augmented_data.append(aug_item)
                    successful += 1
                    
                    if args.show:
                        print(f"\n{'='*60}")
                        print(f"Label: {item['conspiracy']}")
                        print(f"ORIGINAL ({len(item['text'].split())} words):")
                        print(item['text'][:300])
                        print(f"\nREWRITTEN ({len(rewritten.split())} words):")
                        print(rewritten[:300])
                else:
                    failed += 1
                    fail_reasons[reason] += 1
                    
            except Exception as e:
                failed += 1
                fail_reasons[str(e)[:50]] += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if successful + failed > 0:
        print(f"Success rate: {100*successful/(successful+failed):.1f}%")
    print(f"\nFail reasons (top 10):")
    for reason, count in fail_reasons.most_common(10):
        print(f"  {reason}: {count}")
    print(f"\nOriginal samples: {len(data)}")
    print(f"Total samples: {len(augmented_data)}")
    print(f"Distribution: {Counter(d['conspiracy'] for d in augmented_data)}")
    
    # Save
    save_data(augmented_data, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
