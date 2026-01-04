"""
Improved LLM-based data augmentation using Llama-3.2-1B.
Better prompts to ensure paraphrasing, not summarizing.

Key improvements:
1. Explicit instructions to keep SAME LENGTH
2. Instructions to preserve ALL details
3. Few-shot examples showing good paraphrases
4. Temperature tuning for better diversity

Usage:
    python Data_Augmentation/llm_augment_v2.py --sample 20 --show  # Test first
    python Data_Augmentation/llm_augment_v2.py --target Yes No --num_aug 1  # Full run
"""

import json
import argparse
import random
from pathlib import Path
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ============================================
# IMPROVED PROMPTS
# ============================================

PARAPHRASE_PROMPT_V2 = """You are a paraphrasing assistant. Your job is to rewrite text using different words while keeping:
- The EXACT same meaning
- The SAME length (same number of sentences)
- ALL specific details, names, and numbers
- The same tone and style

DO NOT summarize. DO NOT remove any information. Just use different words.

Text to paraphrase: {text}

Paraphrased version:"""


PARAPHRASE_WITH_EXAMPLES = """Rewrite the text using different words. Keep the same length, all details, and same meaning.

Example 1:
Original: The government is hiding evidence from the public about UFO sightings.
Rewritten: Authorities are concealing proof from citizens regarding UFO encounters.

Example 2:
Original: Scientists discovered that the vaccine contains harmful chemicals that cause side effects.
Rewritten: Researchers found that the immunization includes dangerous substances that lead to adverse reactions.

Example 3:
Original: The CEO secretly met with foreign officials to discuss illegal deals.
Rewritten: The chief executive covertly convened with overseas representatives to negotiate unlawful agreements.

Now rewrite this text (keep same length and all details):
Original: {text}
Rewritten:"""


STRICT_PARAPHRASE_PROMPT = """TASK: Rewrite the following text using synonyms and different sentence structures.

RULES:
1. Output must be the SAME LENGTH as input (within 10 words)
2. Keep ALL names, numbers, dates, and specific details
3. Keep the same tone (if suspicious, stay suspicious; if neutral, stay neutral)
4. Do NOT summarize or shorten
5. Do NOT add new information
6. ONLY output the rewritten text, nothing else

INPUT: {text}

OUTPUT:"""


def load_model(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    """Load Llama model."""
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


def paraphrase_v2(model, tokenizer, text, prompt_type="strict", temperature=0.7, max_retries=2):
    """
    Generate paraphrase with improved prompts.
    
    Args:
        prompt_type: "strict", "examples", or "basic"
        temperature: Lower = more conservative, Higher = more creative
        max_retries: Retry if output is too short
    """
    
    # Select prompt
    if prompt_type == "strict":
        prompt = STRICT_PARAPHRASE_PROMPT.format(text=text)
    elif prompt_type == "examples":
        prompt = PARAPHRASE_WITH_EXAMPLES.format(text=text)
    else:
        prompt = PARAPHRASE_PROMPT_V2.format(text=text)
    
    original_word_count = len(text.split())
    
    for attempt in range(max_retries):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(original_word_count * 2) + 50,  # Allow some flexibility
                min_new_tokens=int(original_word_count * 0.5),  # Minimum output length
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Avoid repetition
            )
        
        # Decode
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract paraphrased text
        if "OUTPUT:" in full_output:
            result = full_output.split("OUTPUT:")[-1].strip()
        elif "Rewritten:" in full_output:
            result = full_output.split("Rewritten:")[-1].strip()
        elif "Paraphrased version:" in full_output:
            result = full_output.split("Paraphrased version:")[-1].strip()
        else:
            result = full_output[len(prompt):].strip()
        
        # Clean up
        result = result.split("\n\n")[0].strip()
        result = result.split("Original:")[0].strip()
        result = result.split("INPUT:")[0].strip()
        result = result.split("TASK:")[0].strip()
        
        # Remove quotes if present
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        
        # Validate length
        result_word_count = len(result.split())
        length_ratio = result_word_count / original_word_count if original_word_count > 0 else 0
        
        # Accept if length is reasonable (50% to 150% of original)
        if 0.5 <= length_ratio <= 1.5 and result_word_count >= 10:
            return result
        
        # Retry with lower temperature if too short
        temperature = max(0.3, temperature - 0.2)
    
    # Return None if all retries failed
    return None


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
    parser = argparse.ArgumentParser(description='Improved Llama augmentation')
    parser.add_argument('--input', default='train_rehydrated.jsonl', help='Input file')
    parser.add_argument('--output', default='data/augmented/train_llama_v2.jsonl', help='Output file')
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B-Instruct', help='Model name')
    parser.add_argument('--target', nargs='+', default=['Yes'], help='Labels to augment')
    parser.add_argument('--num_aug', type=int, default=1, help='Augmentations per sample')
    parser.add_argument('--sample', type=int, default=None, help='Sample N items for testing')
    parser.add_argument('--prompt', choices=['strict', 'examples', 'basic'], default='strict',
                        help='Prompt type: strict (best), examples, or basic')
    parser.add_argument('--temperature', type=float, default=0.6, help='Generation temperature')
    parser.add_argument('--exclude', nargs='+', default=["Can't tell"], help='Labels to exclude')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--show', action='store_true', help='Show examples during augmentation')
    
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
    print(f"Prompt type: {args.prompt}")
    print(f"Temperature: {args.temperature}")
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Augment
    augmented_data = data.copy()
    successful = 0
    failed = 0
    
    for item in tqdm(candidates, desc="Augmenting"):
        for aug_idx in range(args.num_aug):
            try:
                aug_text = paraphrase_v2(
                    model, tokenizer, item['text'],
                    prompt_type=args.prompt,
                    temperature=args.temperature
                )
                
                if aug_text:
                    aug_item = {
                        '_id': f"{item['_id']}_llama{aug_idx}",
                        'text': aug_text,
                        'subreddit': item.get('subreddit', ''),
                        'conspiracy': item['conspiracy'],
                        'markers': [],
                        'annotator': item.get('annotator', ''),
                        'augmentation': f'llama_v2_{args.prompt}'
                    }
                    augmented_data.append(aug_item)
                    successful += 1
                    
                    # Show example
                    if args.show:
                        print(f"\n{'='*50}")
                        print(f"ORIGINAL ({len(item['text'].split())} words):")
                        print(item['text'][:200])
                        print(f"\nAUGMENTED ({len(aug_text.split())} words):")
                        print(aug_text[:200])
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"Error: {e}")
                failed += 1
    
    print(f"\n{'='*50}")
    print(f"Augmentation complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {100*successful/(successful+failed):.1f}%")
    print(f"Original samples: {len(data)}")
    print(f"Total samples: {len(augmented_data)}")
    print(f"Distribution: {Counter(d['conspiracy'] for d in augmented_data)}")
    
    # Save
    save_data(augmented_data, args.output)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
