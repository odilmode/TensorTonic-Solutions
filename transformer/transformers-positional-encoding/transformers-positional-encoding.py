import numpy as np
import math

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings using NumPy.
    """
    # 1. Initialize the encoding matrix with zeros
    pe = np.zeros((seq_length, d_model))
    
    # 2. Create the column vector for positions: [0, 1, 2, ..., seq_length-1]
    # Shape becomes (seq_length, 1)
    pos_idx = np.arange(seq_length).reshape(-1, 1)
    
    # 3. Compute the division term (the frequencies)
    # Use np.arange for the 2i indices: [0, 2, 4, ..., d_model-2]
    div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    
    # 4. Apply sine to even indices (0, 2, 4...) and cosine to odd indices (1, 3, 5...)
    # pos_idx * div_term results in a matrix of shape (seq_length, d_model/2)
    pe[:, 0::2] = np.sin(pos_idx * div_term)
    pe[:, 1::2] = np.cos(pos_idx * div_term)
    
    return pe