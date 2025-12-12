#!/usr/bin/env python3
"""
Main augmentation script for conspiracy detection dataset.

Usage:
    # Augment only "Yes" class with 2 augmentations per sample
    python3 augmentation/augment.py --method eda --target Yes --num_aug 2
    
    # Augment all classes (excluding "Can't tell")
    python3 augmentation/augment.py --method eda --num_aug 4
    
    # Preview augmentations without saving
    python3 augmentation/augment.py --method eda --preview 5
"""

import json
import argparse
from pathlib import Path
from collections import Counter

from eda import eda_augment


def load_dataset(path: str, exclude_labels: list[str] = None) -> list[dict]:
    """Load JSONL dataset, optionally excluding certain labels."""
    data = []
    exclude_labels = exclude_labels or []
    
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if item['conspiracy'] not in exclude_labels:
                data.append(item)
    
    return data


def save_dataset(data: list[dict], path: str) -> None:
    """Save dataset to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def augment_dataset(
    data: list[dict],
    method: str = 'eda',
    target_labels: list[str] = None,
    num_aug: int = 4,
    **kwargs
) -> list[dict]:
    """
    Augment dataset with specified method.
    
    Args:
        data: List of data items
        method: Augmentation method ('eda', 'backtranslation', etc.)
        target_labels: Only augment samples with these labels (None = all)
        num_aug: Number of augmentations per sample
        **kwargs: Additional arguments for augmentation method
    
    Returns:
        List containing original + augmented samples
    """
    augmented_data = []
    
    for item in data:
        # Keep original
        augmented_data.append(item)
        
        # Skip if not in target labels
        if target_labels and item['conspiracy'] not in target_labels:
            continue
        
        # Generate augmentations based on method
        if method == 'eda':
            aug_texts = eda_augment(item['text'], num_aug=num_aug, **kwargs)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
        
        # Create augmented samples
        for i, aug_text in enumerate(aug_texts):
            aug_item = {
                '_id': f"{item['_id']}_aug{i}",
                'text': aug_text,
                'subreddit': item['subreddit'],
                'conspiracy': item['conspiracy'],
                'markers': [],  # Drop markers for augmented samples
                'annotator': item['annotator'],
                'augmentation': method  # Track augmentation method
            }
            augmented_data.append(aug_item)
    
    return augmented_data


def preview_augmentations(data: list[dict], method: str, n: int = 5, **kwargs) -> None:
    """Preview augmentations on a few samples."""
    print(f"\n{'='*60}")
    print(f"Preview: {method.upper()} augmentation")
    print(f"{'='*60}\n")
    
    for item in data[:n]:
        print(f"Label: {item['conspiracy']}")
        print(f"Original:\n  {item['text'][:200]}...")
        
        if method == 'eda':
            aug_texts = eda_augment(item['text'], num_aug=2, **kwargs)
        
        for i, aug in enumerate(aug_texts):
            print(f"Aug {i+1}:\n  {aug[:200]}...")
        
        print("-" * 60 + "\n")


def print_stats(original: list[dict], augmented: list[dict]) -> None:
    """Print dataset statistics."""
    orig_counts = Counter(d['conspiracy'] for d in original)
    aug_counts = Counter(d['conspiracy'] for d in augmented)
    
    print("\n" + "="*40)
    print("Dataset Statistics")
    print("="*40)
    print(f"\nOriginal distribution:")
    for label, count in sorted(orig_counts.items()):
        print(f"  {label}: {count}")
    print(f"  Total: {len(original)}")
    
    print(f"\nAugmented distribution:")
    for label, count in sorted(aug_counts.items()):
        print(f"  {label}: {count}")
    print(f"  Total: {len(augmented)}")
    
    new_samples = len(augmented) - len(original)
    print(f"\nNew samples added: {new_samples}")


def main():
    parser = argparse.ArgumentParser(
        description='Data augmentation for conspiracy detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Augment "Yes" class only
  python3 augment.py --method eda --target Yes --num_aug 2
  
  # Augment both Yes and No
  python3 augment.py --method eda --target Yes No --num_aug 4
  
  # Preview without saving
  python3 augment.py --method eda --preview 3
        """
    )
    
    parser.add_argument(
        '--input', 
        default='train_rehydrated.jsonl',
        help='Input JSONL file (default: train_rehydrated.jsonl)'
    )
    parser.add_argument(
        '--output', 
        default='data/augmented/train_augmented.jsonl',
        help='Output JSONL file (default: data/augmented/train_augmented.jsonl)'
    )
    parser.add_argument(
        '--method',
        choices=['eda'],  # Add more methods here later
        default='eda',
        help='Augmentation method (default: eda)'
    )
    parser.add_argument(
        '--target',
        nargs='+',
        default=None,
        help='Labels to augment (default: all). Example: --target Yes'
    )
    parser.add_argument(
        '--num_aug',
        type=int,
        default=4,
        help='Number of augmentations per sample (default: 4)'
    )
    parser.add_argument(
        '--exclude',
        nargs='+',
        default=["Can't tell"],
        help="Labels to exclude from dataset (default: \"Can't tell\")"
    )
    parser.add_argument(
        '--preview',
        type=int,
        default=0,
        help='Preview N samples without saving (default: 0 = disabled)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    # EDA-specific parameters
    parser.add_argument('--alpha_sr', type=float, default=0.1, help='EDA: synonym replacement rate')
    parser.add_argument('--alpha_ri', type=float, default=0.1, help='EDA: random insertion rate')
    parser.add_argument('--alpha_rs', type=float, default=0.1, help='EDA: random swap rate')
    parser.add_argument('--p_rd', type=float, default=0.1, help='EDA: random deletion probability')
    
    args = parser.parse_args()
    
    # Set random seed
    import random
    random.seed(args.seed)
    
    # Resolve paths relative to repo root
    repo_root = Path(__file__).parent.parent
    input_path = repo_root / args.input
    output_path = repo_root / args.output
    
    print(f"Loading data from: {input_path}")
    data = load_dataset(input_path, exclude_labels=args.exclude)
    print(f"Loaded {len(data)} samples (excluded: {args.exclude})")
    
    # Preview mode
    if args.preview > 0:
        preview_augmentations(
            data, 
            args.method, 
            n=args.preview,
            alpha_sr=args.alpha_sr,
            alpha_ri=args.alpha_ri,
            alpha_rs=args.alpha_rs,
            p_rd=args.p_rd
        )
        return
    
    # Augment
    print(f"\nAugmenting with method: {args.method}")
    print(f"Target labels: {args.target or 'all'}")
    print(f"Augmentations per sample: {args.num_aug}")
    
    augmented = augment_dataset(
        data,
        method=args.method,
        target_labels=args.target,
        num_aug=args.num_aug,
        alpha_sr=args.alpha_sr,
        alpha_ri=args.alpha_ri,
        alpha_rs=args.alpha_rs,
        p_rd=args.p_rd
    )
    
    # Print stats
    print_stats(data, augmented)
    
    # Save
    save_dataset(augmented, output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()