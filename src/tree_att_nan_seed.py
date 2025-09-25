import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml





'''
Tree attention with usual cosine positional encoding (PE) added to input embeddings.
'''

class TransformerSelfAttentionLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, seed=None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Set seed for reproducible weight initialization
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            try:
                if torch.backends.mps.is_available():
                    torch.mps.manual_seed(seed)
            except (AttributeError, RuntimeError):
                # torch.mps.manual_seed might not exist in this PyTorch version
                pass
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q1 = nn.Linear(d_model, d_model)
        self.W_q2 = nn.Linear(d_model, d_model)
        self.W_q3 = nn.Linear(d_model, d_model)
        self.W_v2 = nn.Linear(d_model, d_model)
        self.W_v3 = nn.Linear(d_model, d_model)
        
        # Different initialization distributions for W_q1 and W_q2
        # W_q1: Normal distribution
        if seed is not None:
            # Use generator for reproducible initialization
            generator = torch.Generator()
            generator.manual_seed(seed)
            nn.init.normal_(self.W_q1.weight, mean=0.0, std=0.01, generator=generator)
            generator.manual_seed(seed + 1)
            nn.init.uniform_(self.W_q2.weight, a=-0.01, b=0.01, generator=generator)
        else:
            nn.init.normal_(self.W_q1.weight, mean=0.0, std=0.01)
            nn.init.uniform_(self.W_q2.weight, a=-0.01, b=0.01)
        
        nn.init.zeros_(self.W_q1.bias)
        nn.init.zeros_(self.W_q2.bias)
        
        # Initialize remaining layers with seed if provided
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed + 2)
            nn.init.normal_(self.W_q3.weight, mean=0.0, std=0.01, generator=generator)
            generator.manual_seed(seed + 3)
            nn.init.normal_(self.W_v2.weight, mean=0.0, std=0.01, generator=generator)
            generator.manual_seed(seed + 4)
            nn.init.normal_(self.W_v3.weight, mean=0.0, std=0.01, generator=generator)
        else:
            nn.init.normal_(self.W_q3.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.W_v2.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.W_v3.weight, mean=0.0, std=0.01)
        
        nn.init.zeros_(self.W_q3.bias)
        nn.init.zeros_(self.W_v2.bias)
        nn.init.zeros_(self.W_v3.bias)
        
        self.W_o = nn.Linear(d_model, d_model) if n_heads > 1 else None
        if self.W_o is not None:
            if seed is not None:
                generator = torch.Generator()
                generator.manual_seed(seed + 5)
                nn.init.normal_(self.W_o.weight, mean=0.0, std=0.01, generator=generator)
            else:
                nn.init.normal_(self.W_o.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.W_o.bias)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.LeakyReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Initialize FFN layers with seed if provided
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed + 6)
            nn.init.normal_(self.ffn[0].weight, mean=0.0, std=0.01, generator=generator)
            nn.init.zeros_(self.ffn[0].bias)
            generator.manual_seed(seed + 7)
            nn.init.normal_(self.ffn[2].weight, mean=0.0, std=0.01, generator=generator)
            nn.init.zeros_(self.ffn[2].bias)
        else:
            nn.init.normal_(self.ffn[0].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.ffn[0].bias)
            nn.init.normal_(self.ffn[2].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.ffn[2].bias)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def scaled_dot_product_attention(self, Q1, Q2, Q3, V2, V3, mask=None):
        # Handle both [heads, seq_len, d_k] and [batch, heads, seq_len, d_k]
        if Q1.dim() == 3:
            # Single sequence: [heads, seq_len, d_k]
            batch_size = None
            n_heads, seq_len, d_k = Q1.size()
        else:
            # Batch: [batch, heads, seq_len, d_k]
            batch_size, n_heads, seq_len, d_k = Q1.size()

        # Debug: Check inputs for NaN
        # if torch.isnan(Q1).any():
        #     print("NaN detected in Q1!")
        # if torch.isnan(Q2).any():
        #     print("NaN detected in Q2!")
        # if torch.isnan(Q3).any():
        #     print("NaN detected in Q3!")
        # if torch.isnan(V2).any():
        #     print("NaN detected in V2!")
        # if torch.isnan(V3).any():
        #     print("NaN detected in V3!")

        # Compute attention scores
        scores12 = torch.matmul(Q1, Q2.transpose(-2, -1)) / (2*math.sqrt(self.d_k))
        scores23 = torch.matmul(Q2, Q3.transpose(-2, -1)) / (2*math.sqrt(self.d_k))

        scores12 = scores12.clamp(max=20, min=-20)
        scores23 = scores23.clamp(max=20, min=-20)

        # Debug: Check scores for NaN
        # if torch.isnan(scores12).any():
        #     print("NaN detected in scores12!")
        #     print(f"Q1 stats: min={Q1.min()}, max={Q1.max()}, mean={Q1.mean()}")
        #     print(f"Q2 stats: min={Q2.min()}, max={Q2.max()}, mean={Q2.mean()}")
        # if torch.isnan(scores23).any():
        #     print("NaN detected in scores23!")
        #     print(f"Q2 stats: min={Q2.min()}, max={Q2.max()}, mean={Q2.mean()}")
        #     print(f"Q3 stats: min={Q3.min()}, max={Q3.max()}, mean={Q3.mean()}")
        
        if mask is not None:
            if batch_size is not None and mask.dim() == 2:
                # Expand mask for batch dimension
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            elif batch_size is None and mask.dim() == 2:
                # Single sequence mask
                mask = mask.unsqueeze(0)  # [1, seq_len, seq_len]
            scores12 = scores12.masked_fill(mask == 0, -1e9)
            scores23 = scores23.masked_fill(mask == 0, -1e9)

        eQ12 = torch.exp(scores12)
        eQ23 = torch.exp(scores23)



        
        # Debug: Check exponential values for NaN/Inf
        # if torch.isnan(eQ12).any():
        #     print("NaN detected in eQ12!")
        #     print(f"scores12 stats: min={scores12.min()}, max={scores12.max()}, mean={scores12.mean()}")
        # if torch.isinf(eQ12).any():
        #     print("Inf detected in eQ12!")
        #     print(f"scores12 stats: min={scores12.min()}, max={scores12.max()}, mean={scores12.mean()}")
        # if torch.isnan(eQ23).any():
        #     print("NaN detected in eQ23!")
        #     print(f"scores23 stats: min={scores23.min()}, max={scores23.max()}, mean={scores23.mean()}")
        # if torch.isinf(eQ23).any():
        #     print("Inf detected in eQ23!")
        #     print(f"scores23 stats: min={scores23.min()}, max={scores23.max()}, mean={scores23.mean()}")
        
        # Computing denominator of softmax
        sum_eQ23 = eQ23.sum(dim=-1, keepdim=True)
        R = torch.max(torch.matmul(eQ12, sum_eQ23), torch.tensor(1e-9))  # Prevent division by zero

        # Debug: Check sum and R for NaN/Inf
        # if torch.isnan(sum_eQ23).any():
        #     print("NaN detected in sum_eQ23!")
        # if torch.isinf(sum_eQ23).any():
        #     print("Inf detected in sum_eQ23!")
        # if torch.isnan(R).any():
        #     print("NaN detected in R!")
        # if torch.isinf(R).any():
        #     print("Inf detected in R!")
        # if (R == 0).any():
        #     print("Zero values detected in R!")
        #     print(f"R stats: min={R.min()}, max={R.max()}, mean={R.mean()}")

        # Initialize P23 tensor with correct dimensions
        if batch_size is not None:
            P23 = torch.zeros(batch_size, n_heads, seq_len, d_k, dtype=torch.float32, device=Q1.device)
        else:
            P23 = torch.zeros(n_heads, seq_len, d_k, dtype=torch.float32, device=Q1.device)
        
        # Keep the original for loop intact
        for d in range(self.d_k):
            if batch_size is not None:
                # Batch processing
                V2_d = V2[:, :, :, d].unsqueeze(-1)  # [batch, heads, seq_len, 1]
                eQ12V = eQ12 * V2_d  # [batch, heads, seq_len, seq_len]
                V3_d = V3[:, :, :, d].unsqueeze(-1)  # [batch, heads, seq_len, 1]
                eQ23V = torch.matmul(eQ23, V3_d)  # [batch, heads, seq_len, 1]
                P = torch.matmul(eQ12V, eQ23V)  # [batch, heads, seq_len, 1]
                P23[:, :, :, d] = P.squeeze(-1)  # [batch, heads, seq_len]
            else:
                # Single sequence processing (original code)
                V2_d = V2[:, :, d].unsqueeze(-1)  # [heads, seq_len, 1]
                eQ12V = eQ12 * V2_d  # [heads, seq_len, seq_len]
                V3_d = V3[:, :, d].unsqueeze(-1)  # [heads, seq_len, 1]
                eQ23V = torch.matmul(eQ23, V3_d)  # [heads, seq_len, 1]
                P = torch.matmul(eQ12V, eQ23V)  # [heads, seq_len, 1]
                P23[:, :, d] = P.squeeze(-1)  # [heads, seq_len]
        
        # Debug: Check P23 for NaN
        # if torch.isnan(P23).any():
        #     print("NaN detected in P23!")
        
        result = P23 / R
        
        # Debug: Check final result for NaN
        # if torch.isnan(result).any():
        #     print("NaN detected in final result!")
        #     print(f"P23 stats: min={P23.min()}, max={P23.max()}, mean={P23.mean()}")
        #     print(f"R stats: min={R.min()}, max={R.max()}, mean={R.mean()}")
        
        return result





    def split_heads(self, x):
        # x: [batch, seq_len, d_model] or [seq_len, d_model]
        if x.dim() == 2:
            # Single sequence: [seq_len, d_model]
            L, _ = x.size()
            x = x.view(L, self.n_heads, self.d_k)  # [seq_len, heads, d_k]
            return x.permute(1, 0, 2)  # [heads, seq_len, d_k]
        else:
            # Batch of sequences: [batch, seq_len, d_model]
            batch_size, L, _ = x.size()
            x = x.view(batch_size, L, self.n_heads, self.d_k)  # [batch, seq_len, heads, d_k]
            return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, d_k]

    def combine_heads(self, x):
        if x.dim() == 3:
            # Single sequence: [heads, seq_len, d_k]
            _, L, _ = x.size()
            x = x.permute(1, 0, 2).contiguous()  # [seq_len, heads, d_k]
            return x.view(L, self.d_model)  # [seq_len, d_model]
        else:
            # Batch of sequences: [batch, heads, seq_len, d_k]
            batch_size, _, L, _ = x.size()
            x = x.permute(0, 2, 1, 3).contiguous()  # [batch, seq_len, heads, d_k]
            return x.view(batch_size, L, self.d_model)  # [batch, seq_len, d_model]

    def forward(self, x, mask=None):
        # x: [batch, seq_len, d_model] or [seq_len, d_model]
        input_was_2d = (x.dim() == 2)
        
        Q1 = self.split_heads(self.W_q1(x))
        Q2 = self.split_heads(self.W_q2(x))
        Q3 = self.split_heads(self.W_q3(x))
        V2 = self.split_heads(self.W_v2(x))
        V3 = self.split_heads(self.W_v3(x))

        attn_output = self.scaled_dot_product_attention(Q1, Q2, Q3, V2, V3, mask)
        attn_output = self.combine_heads(attn_output)
        
        if self.W_o is not None:
            attn_output = self.W_o(attn_output)
            
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x



class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=5000, device='mps'):
        super().__init__()
        # Set default device to MPS if available, fallback to CPU
        if device == 'mps' and not torch.backends.mps.is_available():
            device = 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        
        self.device = torch.device(device)
        
        pe = torch.zeros(max_len, d_model, dtype=torch.float32, device=self.device)
        position = torch.arange(0, max_len, dtype=torch.float32, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=self.device) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: [1, max_len, d_model]

    def forward(self, x):
        # x: [batch, seq_len, d_model] (expects batch dimension)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]





class Transformer(nn.Module):
    def __init__(self, config_path, device='mps', seed=None):
        super().__init__()
        
        # Set seed for reproducible model initialization
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            try:
                if torch.backends.mps.is_available():
                    torch.mps.manual_seed(seed)
            except (AttributeError, RuntimeError):
                # torch.mps.manual_seed might not exist in this PyTorch version
                pass
        
        # Set default device to MPS if available, fallback to CPU
        if device == 'mps' and not torch.backends.mps.is_available():
            device = 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            
        self.device = torch.device(device)
        self.config = self.load_config(config_path)
        self.layers = nn.ModuleList()
        
        first_d_model = self.config['layers'][0].get('d_model', 512)
        self.pos_encoder = PositionalEncoding(first_d_model, device=device)

        for i, layer_config in enumerate(self.config['layers']):
            # Use different seed offsets for each layer if seed is provided
            layer_seed = seed + i if seed is not None else None
            layer = TransformerSelfAttentionLayer(
                d_model=layer_config.get('d_model', 512),
                n_heads=layer_config.get('n_heads', 8),
                d_ff=layer_config.get('d_ff', 2048),
                seed=layer_seed
            )
            self.layers.append(layer)
        
        # Move entire model to device
        self.to(self.device)
        
        # Classification head: 1 logit per token (commented out to match transformer.py behavior)
        # self.classifier = nn.Linear(first_d_model, 1)

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def forward(self, x, mask=None):
        # x: [batch, seq_len, d_model] or [seq_len, d_model]
        input_was_2d = (x.dim() == 2)
        
        if input_was_2d:
            # Single sequence: add batch dimension
            x = x.unsqueeze(0)  # [1, seq_len, d_model]
        elif x.dim() != 3:
            raise ValueError(f"Expected input of shape [seq_len, d_model] or [batch, seq_len, d_model], got {x.shape}")

        # Ensure input is on the same device as the model
        x = x.to(self.device)
        
        # Apply positional encoding
        x = self.pos_encoder(x)  # expects [batch, seq_len, d_model]
        
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        # Remove batch dimension if input was 2D
        if input_was_2d:
            x = x.squeeze(0)  # [seq_len, d_model]
        
        return x

    def get_layer_info(self):
        print(f"Transformer with {len(self.layers)} layers:")
        for i, layer_config in enumerate(self.config['layers']):
            print(f"Layer {i+1}: d_model={layer_config.get('d_model', 512)}, "
                  f"n_heads={layer_config.get('n_heads', 8)}, "
                  f"d_ff={layer_config.get('d_ff', 2048)}")


def create_sample_config(config_path="transformer_config.yaml"):
    sample_config = {
        'model_name': 'ConfigurableTransformer',
        'layers': [
            {'d_model': 512, 'n_heads': 8, 'd_ff': 2048},
            {'d_model': 512, 'n_heads': 8, 'd_ff': 2048},
            {'d_model': 512, 'n_heads': 8, 'd_ff': 2048}
        ]
    }
    with open(config_path, 'w') as file:
        yaml.dump(sample_config, file, default_flow_style=False, indent=2)
    print(f"Sample configuration saved to {config_path}")
    return config_path

