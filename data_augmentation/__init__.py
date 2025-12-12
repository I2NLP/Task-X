"""
Data augmentation module for conspiracy detection.

Available methods:
- EDA (Easy Data Augmentation): eda.py
"""

from .eda import (
    eda_augment,
    eda_augment_all,
    synonym_replacement,
    random_insertion,
    random_swap,
    random_deletion
)

__all__ = [
    'eda_augment',
    'eda_augment_all',
    'synonym_replacement',
    'random_insertion',
    'random_swap',
    'random_deletion'
]