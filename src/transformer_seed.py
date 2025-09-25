import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml


class TransformerSelfAttentionLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, tr=1, d_ff=2048, seed=None):
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
        self.tr = tr

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Initialize linear layers with seed if provided
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            nn.init.xavier_uniform_(self.W_q.weight, generator=generator)
            nn.init.zeros_(self.W_q.bias)
            generator.manual_seed(seed + 1)
            nn.init.xavier_uniform_(self.W_k.weight, generator=generator)
            nn.init.zeros_(self.W_k.bias)
            generator.manual_seed(seed + 2)
            nn.init.xavier_uniform_(self.W_v.weight, generator=generator)
            nn.init.zeros_(self.W_v.bias)
        else:
            nn.init.xavier_uniform_(self.W_q.weight)
            nn.init.zeros_(self.W_q.bias)
            nn.init.xavier_uniform_(self.W_k.weight)
            nn.init.zeros_(self.W_k.bias)
            nn.init.xavier_uniform_(self.W_v.weight)
            nn.init.zeros_(self.W_v.bias)
        
        self.W_o = nn.Linear(d_model, d_model) if n_heads > 1 else None
        if self.W_o is not None:
            if seed is not None:
                generator = torch.Generator()
                generator.manual_seed(seed + 3)
                nn.init.xavier_uniform_(self.W_o.weight, generator=generator)
            else:
                nn.init.xavier_uniform_(self.W_o.weight)
            nn.init.zeros_(self.W_o.bias)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Initialize FFN layers with seed if provided
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed + 4)
            nn.init.xavier_uniform_(self.ffn[0].weight, generator=generator)
            nn.init.zeros_(self.ffn[0].bias)
            generator.manual_seed(seed + 5)
            nn.init.xavier_uniform_(self.ffn[2].weight, generator=generator)
            nn.init.zeros_(self.ffn[2].bias)
        else:
            nn.init.xavier_uniform_(self.ffn[0].weight)
            nn.init.zeros_(self.ffn[0].bias)
            nn.init.xavier_uniform_(self.ffn[2].weight)
            nn.init.zeros_(self.ffn[2].bias)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = self.tr * torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, V)

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
        # x: [seq_len, d_model] or [batch, seq_len, d_model]
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
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
                tr=layer_config.get('tr', 1),
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
        
        # x should be [seq_len, d_model] or [batch, seq_len, d_model]
        if x.dim() == 2:
            # Single sequence case: [seq_len, d_model]
            x = x.to(self.device)
            x = self.pos_encoder(x.unsqueeze(0)).squeeze(0)  # Add batch dim for pos_encoder, then remove
        elif x.dim() == 3:
            # Batch case: [batch, seq_len, d_model]
            x = x.to(self.device)
            x = self.pos_encoder(x)  # pos_encoder expects batch dimension
        else:
            raise ValueError(f"Expected input of shape [seq_len, d_model] or [batch, seq_len, d_model], got {x.shape}")
        
        x = x.to(self.device)
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        return x  # [seq_len, d_model] or [batch, seq_len, d_model]

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
