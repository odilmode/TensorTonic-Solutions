import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.spec_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        for id, word in enumerate(self.spec_tokens):
            self.word_to_id[word] = id

        unique_ = set()
        for text in texts:
            t = text.split()
            unique_.update(t)

        current_id = len(self.spec_tokens)
        for word in sorted(list(unique_)):
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
            current_id += 1
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        words= text.split()
        unk_id = self.word_to_id.get("<UNK>")
        return [self.word_to_id.get(word, unk_id) for word in words]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        words = [self.id_to_word.get(id, "<UNK>") for id in ids]
        return " ".join(words)
