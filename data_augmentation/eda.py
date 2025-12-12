"""
Methods:
- Synonym Replacement (SR)
- Random Insertion (RI)
- Random Swap (RS)
- Random Deletion (RD)
"""

import random
from nltk.corpus import wordnet
import nltk

# Download WordNet if needed
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def get_synonyms(word: str) -> list[str]:
    """Get synonyms for a word from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym)
    return list(synonyms)


def synonym_replacement(words: list[str], n: int) -> list[str]:
    """Replace n random words with their synonyms."""
    new_words = words.copy()
    random_word_indices = list(range(len(words)))
    random.shuffle(random_word_indices)
    
    num_replaced = 0
    for idx in random_word_indices:
        word = words[idx]
        synonyms = get_synonyms(word)
        if synonyms:
            new_words[idx] = random.choice(synonyms)
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words


def random_insertion(words: list[str], n: int) -> list[str]:
    """Insert n random synonyms at random positions."""
    new_words = words.copy()
    for _ in range(n):
        _add_word(new_words)
    return new_words


def _add_word(words: list[str]) -> None:
    """Helper: insert a synonym of a random word at a random position."""
    if not words:
        return
    for _ in range(10):  # try up to 10 times to find a word with synonyms
        random_word = random.choice(words)
        synonyms = get_synonyms(random_word)
        if synonyms:
            words.insert(random.randint(0, len(words)), random.choice(synonyms))
            return


def random_swap(words: list[str], n: int) -> list[str]:
    """Swap n pairs of words randomly."""
    new_words = words.copy()
    for _ in range(n):
        if len(new_words) >= 2:
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words


def random_deletion(words: list[str], p: float) -> list[str]:
    """Delete each word with probability p."""
    if len(words) == 1:
        return words
    new_words = [w for w in words if random.random() > p]
    # Ensure at least one word remains
    return new_words if new_words else [random.choice(words)]


def eda_augment(
    text: str,
    alpha_sr: float = 0.1,
    alpha_ri: float = 0.1,
    alpha_rs: float = 0.1,
    p_rd: float = 0.1,
    num_aug: int = 4
) -> list[str]:
    words = text.split()
    num_words = len(words)
    
    if num_words == 0:
        return []
    
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))
    
    augmented = []
    
    for _ in range(num_aug):
        # Randomly choose one augmentation type
        aug_type = random.choice(['sr', 'ri', 'rs', 'rd'])
        
        if aug_type == 'sr':
            new_words = synonym_replacement(words, n_sr)
        elif aug_type == 'ri':
            new_words = random_insertion(words, n_ri)
        elif aug_type == 'rs':
            new_words = random_swap(words, n_rs)
        else:
            new_words = random_deletion(words, p_rd)
        
        augmented.append(' '.join(new_words))
    
    return augmented


def eda_augment_all(
    text: str,
    alpha_sr: float = 0.1,
    alpha_ri: float = 0.1,
    alpha_rs: float = 0.1,
    p_rd: float = 0.1
) -> dict[str, str]:
    
    words = text.split()
    num_words = len(words)
    
    if num_words == 0:
        return {}
    
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))
    
    return {
        'sr': ' '.join(synonym_replacement(words, n_sr)),
        'ri': ' '.join(random_insertion(words, n_ri)),
        'rs': ' '.join(random_swap(words, n_rs)),
        'rd': ' '.join(random_deletion(words, p_rd))
    }