import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
import time
import numpy as np
import os

from tree_att_nan_seed import TransformerSelfAttentionLayer, Transformer
# from transformer_seed import TransformerSelfAttentionLayer, Transformer

import random

def create_test_config(d_model=16, d_ff=256, n_heads=8):
    """Create a temporary config file for testing"""
    config = {
        'model_name': 'TestTransformer',
        'layers': [
            {'d_model': d_model, 'n_heads': n_heads, 'd_ff': d_ff},
            {'d_model': d_model, 'n_heads': n_heads, 'd_ff': d_ff}
        ]
    }
    
    config_path = "test_transformer_config.yaml"
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)
    
    return config_path

if __name__ == "__main__":
    # Parameters
    d_model = 16
    d_ff = 256
    batch_size = 1024
    seq_len = 200
    n_heads = 8
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"Creating transformer with d_model={d_model}, d_ff={d_ff}, n_heads={n_heads}")
    print(f"Input: batch_size={batch_size}, seq_len={seq_len}")
    
    # Create config and model
    config_path = create_test_config(d_model=d_model, d_ff=d_ff, n_heads=n_heads)
    
    try:
        # Create transformer model
        model = Transformer(config_path, device=device, seed=42)
        model.eval()
        elapsed_time = 0.0
        for i in range(10):
        # Generate 10 random input
            input_data = torch.randn(batch_size, seq_len, d_model)
            
            # print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            # print(f"Input shape: {input_data.shape}")
            
            # Forward pass with timing
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = model(input_data)
            
            end_time = time.perf_counter()
            elapsed_time += end_time - start_time
        elapsed_time /= 10  # Average over 10 runs
        print(f"Output shape: {output.shape}")
        print(f"Forward pass time: {elapsed_time*1000:.3f} ms")
        print(f"Tokens/second: {(batch_size * seq_len) / elapsed_time:.1f}")
        
    finally:
        # Clean up config file
        if os.path.exists(config_path):
            os.remove(config_path)
