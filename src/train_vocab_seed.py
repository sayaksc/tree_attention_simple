import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import os
import sys
# Import from tree_att
from tree_att_nan_seed import TransformerSelfAttentionLayer, create_sample_config, Transformer


# Import from self-attention
# from transformer_seed import TransformerSelfAttentionLayer, create_sample_config, Transformer
    
import matplotlib.pyplot as plt
import random
import time


global tree_used 
tree_used = False
#If imported from tree_att... then tree_used = True
if any(module.startswith('tree_att') for module in sys.modules):
    tree_used = True
    print("Using tree_att module", "="*50)


def setup_device(device_name='mps'):
    """
    Setup device with MPS as default, fallback to CPU if MPS/CUDA unavailable
    
    Args:
        device_name: Preferred device ('mps', 'cuda', 'cpu')
    
    Returns:
        torch.device: The available device
    """
    if device_name == 'mps' and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    elif device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device (requested: {device_name})")
    
    return device


def set_seed(seed, verbose=False):
    """
    Set random seed for all relevant random number generators for reproducibility
    
    Args:
        seed (int): Random seed to set
        verbose (bool): Whether to print seed information
    """
    if seed is None:
        return
        
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    np.random.seed(seed)
    random.seed(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Additional GPU determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if verbose:
        print(f"Random seed set to: {seed}")



def set_global_seed(seed):
    """
    Set global random seed for all random number generators.
    This ensures reproducible results across torch, numpy, random, and CUDA.
    
    Args:
        seed (int): Random seed to set globally
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
        torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
        # MPS seeding might not be available in all PyTorch versions
        try:
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)
        except (AttributeError, RuntimeError):
            # torch.mps.manual_seed might not exist in this PyTorch version
            pass
        np.random.seed(seed)
        random.seed(seed)
        # For deterministic behavior (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Additional determinism
        os.environ['PYTHONHASHSEED'] = str(seed)
        print(f"Global random seed set to: {seed}")

class VocabTransformer(nn.Module):
    """
    Transformer wrapper that processes one-hot vocabulary inputs and produces vocab projection with softmax at output
    """
    def __init__(self, base_transformer, vocab_size, d_model=None, seed=None):
        super(VocabTransformer, self).__init__()
        
        # Set seed for VocabTransformer initialization if provided
        if seed is not None:
            torch.manual_seed(seed + 1000)  # Offset to avoid conflicts with base transformer
            torch.cuda.manual_seed(seed + 1000) if torch.cuda.is_available() else None
            torch.cuda.manual_seed_all(seed + 1000) if torch.cuda.is_available() else None
            np.random.seed(seed + 1000)
            random.seed(seed + 1000)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.base_transformer = base_transformer
        self.vocab_size = vocab_size

        # Get d_model from base transformer config
        if d_model is None:
            d_model = base_transformer.config['layers'][0].get('d_model', 512)
        self.d_model = d_model
        
        # Input vocabulary embedding - maps discrete tokens to continuous embeddings
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        
        # Output vocabulary projection - maps transformer output back to vocabulary logits
        self.output_projection = nn.Linear(d_model, vocab_size)

        # self.output_projection = nn.Sequential(
        #     nn.Linear(d_model, base_transformer.config['layers'][0].get('d_ff', 512)),
        #     nn.ReLU(),
        #     nn.Linear(base_transformer.config['layers'][0].get('d_ff', 512), vocab_size)
        # )
        
        # Initialize embeddings with proper scaling and seeded generators if seed provided
        if seed is not None:
            # Use generator for reproducible initialization
            generator = torch.Generator()
            generator.manual_seed(seed + 1000)
            nn.init.normal_(self.input_embedding.weight, mean=0, std=0.02, generator=generator)
            generator.manual_seed(seed + 1001)
            nn.init.normal_(self.output_projection.weight, mean=0, std=0.02, generator=generator)
        else:
            nn.init.normal_(self.input_embedding.weight, mean=0, std=0.02)
            nn.init.normal_(self.output_projection.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
        
        print(f"VocabTransformer created: vocab_size={vocab_size}, d_model={d_model}")
        if seed is not None:
            print(f"VocabTransformer initialized with seed offset: {seed + 1000}")
    
    def forward(self, x, mask=None):
        """
        Forward pass with one-hot vocabulary input and softmax output
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, vocab_size] (one-hot encodings)
            mask: Optional attention mask
        
        Returns:
            Output probabilities of shape [batch_size, seq_len, vocab_size] after softmax
        """
        # Handle one-hot encoded inputs
        if x.dim() == 3 and x.size(-1) == self.vocab_size:
            # if torch.isnan(x).any():
            #     print(f"[NaN DETECTED] Input x shape: {x.shape}, min: {x.min():.6f}, max: {x.max():.6f}")
            # if torch.isnan(self.input_embedding.weight).any():
            #     print(f"[NaN DETECTED] Embedding weights min: {self.input_embedding.weight.min():.6f}, max: {self.input_embedding.weight.max():.6f}")
            
            # Convert one-hot to embeddings by matrix multiplication with embedding weights
            # This is equivalent to: embedding_table[token_indices] but works with soft/one-hot inputs
            # x: [batch_size, seq_len, vocab_size] (one-hot vectors)
            # self.input_embedding.weight: [vocab_size, d_model] (embedding table)
            # Result: [batch_size, seq_len, d_model] (embedded representations)
            x = torch.matmul(x, self.input_embedding.weight)  # [batch_size, seq_len, d_model]
            # if torch.isnan(x).any():
            #     print(f"[NaN DETECTED] After embedding x shape: {x.shape}, min: {x.min():.6f}, max: {x.max():.6f}")
        elif x.dim() == 3 and x.size(-1) == self.d_model:
            # Already continuous embeddings, use as is
            # if torch.isnan(x).any():
            #     print(f"[NaN DETECTED] Input x (already embedded) shape: {x.shape}, min: {x.min():.6f}, max: {x.max():.6f}")
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected [batch_size, seq_len, {self.vocab_size}] for one-hot or [batch_size, seq_len, {self.d_model}] for embeddings.")
        
        # Pass through base transformer
        transformer_output = self.base_transformer(x, mask=mask)  # [batch_size, seq_len, d_model]
        # if torch.isnan(transformer_output).any():
        #     print(f"[NaN DETECTED] Transformer output shape: {transformer_output.shape}, min: {transformer_output.min():.6f}, max: {transformer_output.max():.6f}")
        
        # Project to vocabulary logits
        # if torch.isnan(self.output_projection.weight).any():
        #     print(f"[NaN DETECTED] Output projection weights min: {self.output_projection.weight.min():.6f}, max: {self.output_projection.weight.max():.6f}")
        logits = self.output_projection(transformer_output)  # [batch_size, seq_len, vocab_size]
        # if torch.isnan(logits).any():
        #     print(f"[NaN DETECTED] Raw logits shape: {logits.shape}, min: {logits.min():.6f}, max: {logits.max():.6f}")

        # logits = torch.matmul(transformer_output, self.input_embedding.weight.T)
        
        logits = logits.clamp(max=50)
        # if torch.isnan(logits).any():
        #     print(f"[NaN DETECTED] Clamped logits min: {logits.min():.6f}, max: {logits.max():.6f}")

        logits = logits - torch.max(logits, dim=-1, keepdim=True).values  # For numerical stability
        # if torch.isnan(logits).any():
        #     print(f"[NaN DETECTED] Normalized logits min: {logits.min():.6f}, max: {logits.max():.6f}")
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        # if torch.isnan(probs).any():
        #     print(f"[NaN DETECTED] F.softmax probs min: {probs.min():.6f}, max: {probs.max():.6f}")
        
        probs = torch.exp(logits)/(torch.max(torch.sum(torch.exp(logits), dim=-1, keepdim=True), torch.tensor(1e-10)))
        # if torch.isnan(probs).any():
        #     print(f"[NaN DETECTED] Final probs min: {probs.min():.6f}, max: {probs.max():.6f}")
        
        return probs
    
    def get_layer_info(self):
        """Print information about the model layers"""
        print(f"VocabTransformer:")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Model dimension: {self.d_model}")
        print(f"  Input embedding: {self.input_embedding}")
        print(f"  Output projection: {self.output_projection}")
        print(f"  Base transformer:")
        self.base_transformer.get_layer_info()
    
    @property
    def config(self):
        """Access base transformer config"""
        return self.base_transformer.config
    
    @property
    def layers(self):
        """Access base transformer layers"""
        return self.base_transformer.layers




def function_composition(seq_len, batch_size=32, n=10, x=None, k=2, seed=None, device='mps'):
    """
    Fully vectorized function composition for batch generation.
    Creates token sequences for vocabulary-based transformer training.
    
    Args:
        seq_len: Sequence length
        batch_size: Number of sequences to generate
        n: Range for random indices (0 to n-1)
        x: Starting value (if None, randomly generated for each batch item)
        k: Number of composition steps
        seed: Random seed for reproducibility
        device: Device to place tensors on
    
    Returns:
        batch_inputs: [batch_size, seq_len] - token sequences
        batch_targets: [batch_size] - target values
    """
    if seq_len < k*n + 1:
        raise ValueError(f"seq_len must be at least {k*n + 1} for k={k} and n={n}")
    # if d_model < 3:
    #     raise ValueError("d_model must be at least 3 to accommodate input format")
    
    # Set default device to MPS if available, fallback to CPU
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
        torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
        random.seed(seed) 
        np.random.seed(seed)
        # For deterministic behavior on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Initialize batch tensors - now just 2D for token sequences
    batch_inputs = torch.zeros(batch_size, seq_len, 2, dtype=torch.long, device=device)
    
    # Generate starting values for each batch item
    if x is None:
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            x_values = torch.randint(0, n, (batch_size,), device=device, generator=generator)
        else:
            x_values = torch.randint(0, n, (batch_size,), device=device)
    else:
        x_values = torch.full((batch_size,), x, device=device)
    
    # Generate all function mappings for all batch items at once
    # Shape: [batch_size, k, n] - for each batch item, k functions each mapping n values
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + 1)  # Use different seed for functions
        all_functions = torch.randint(0, n, (batch_size, k, n), device=device, generator=generator)
    else:
        all_functions = torch.randint(0, n, (batch_size, k, n), device=device)

    # Compute function composition results for all batch items
    batch_targets = x_values.clone()
    for step in range(k):
        # For each batch item, apply the step-th function
        indices = batch_targets.long()
        batch_targets = all_functions[torch.arange(batch_size), step, indices]

    # Fill the input sequences with function values only (no positional encoding needed)
    total_positions = k * n
    positions_to_fill = min(total_positions, seq_len)
    
    if positions_to_fill > 0:
        # Create indices for all positions at once
        all_step_indices = torch.arange(positions_to_fill, device=device) // n  # Which step (0, 1, ..., k-1)
        all_pos_indices = torch.arange(positions_to_fill, device=device) % n   # Which position within step (0, 1, ..., n-1)

        # Fill with function values - vectorized
        batch_inputs[:, :positions_to_fill, 0] = all_functions[
            torch.arange(batch_size).unsqueeze(1), 
            all_step_indices.unsqueeze(0), 
            all_pos_indices.unsqueeze(0)
        ]
        batch_inputs[:, :positions_to_fill, 1] = (torch.arange(positions_to_fill, device=device) % n).unsqueeze(0)  # Placeholder for second feature if needed

    # Set the starting value in the last position
    if seq_len > 0:
        batch_inputs[:, -1, 0] = x_values
        batch_inputs[:, -1, 1] = seq_len - 1  # Position index for the last token

    return batch_inputs, batch_targets







def train_transformer(model, device, seq_len, n=10, fold=2, x=None, num_epochs=20, learning_rate=0.001, batch_size=32, mask=None, weight_decay=0.0, seed=None):
    """
    Train transformer model on function composition task with vocabulary support
    
    Args:
        model: Transformer model to train (will be wrapped with VocabTransformer)
        device: Device to run training on (e.g., 'mps', 'cuda', 'cpu')
        seq_len (int): Sequence length
        n (int): Range for function composition (default: 10)
        fold (int): Number of composition steps (default: 2)
        x: Starting value for composition (default: None for random)
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        batch_size (int): Number of instances per batch
        mask: Optional attention mask
        weight_decay (float): L2 regularization strength (default: 0.0)
        seed (int): Random seed for reproducibility (default: None)
    
    Returns:
        VocabTransformer: The trained vocabulary transformer model
    """
    
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
        torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
        np.random.seed(seed)
        random.seed(seed)
        # For deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {seed}")
    
    # Wrap the base transformer with vocabulary support
    vocab_model = VocabTransformer(model, vocab_size=3+n+n, seed=seed).to(device)
    
    print(f"Training on device: {device}")
    vocab_model.get_layer_info()

    # Setup optimizer - ensure seed is set for any random initialization
    if seed is not None:
        torch.manual_seed(seed + 2000)  # Different offset for optimizer
        torch.cuda.manual_seed(seed + 2000) if torch.cuda.is_available() else None
        np.random.seed(seed + 2000)
        random.seed(seed + 2000)
    
    optimizer = torch.optim.Adam(vocab_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(vocab_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    
    print(f"Training with vocab_size={vocab_model.vocab_size}, seq_len={seq_len}, batch_size={batch_size}, device={device}")
    Total_loss = []
    Accuracy = []
    start_time = time.time()
    time_epoch = 0
    prev_time = 0
    Time = []
    
    # Train the transformer for the specified number of epochs
    for epoch in range(num_epochs):
        # Create training data for this batch with epoch-specific seed
        # Use epoch+1 multiplied by a large number to avoid seed collisions
        epoch_seed = (seed * 10000 + epoch + 1) if seed is not None else None
        batch_inputs, batch_targets = function_composition(seq_len, batch_size=batch_size, n=n, k=fold, x=x, device=device, seed=epoch_seed)


        prev_time = time.time()

        # Ensure deterministic behavior for this batch if seed is provided
        if seed is not None:
            torch.manual_seed(seed * 10000 + epoch + 1)  # Same as epoch_seed
            torch.cuda.manual_seed(seed * 10000 + epoch + 1) if torch.cuda.is_available() else None

        # Move batch data to device
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        # Convert batch_inputs to one-hot encodings
        batch_inputs_one_hot1 = F.one_hot(batch_inputs[:,:, 0], num_classes=n).float().to(device)
        batch_inputs_one_hot2 = F.one_hot(batch_inputs[:,:, 1], num_classes=n).float().to(device)
        batch_inputs_one_hot = torch.cat([batch_inputs_one_hot1, batch_inputs_one_hot2], dim=-1).to(device)

        batch_inputs_one_hot = torch.cat([torch.zeros(batch_size, seq_len, 3, device=device), batch_inputs_one_hot], dim=-1).to(device)
        batch_inputs_one_hot[:, :n, 0] = 1.0  # Ensure padding token is all zeros except first index
        batch_inputs_one_hot[:, n:2*n, 1] = 1.0  # Ensure padding token is all zeros except second index
        batch_inputs_one_hot[:, 2*n, 2] = 1.0  # Ensure padding token is all zeros except third index
        
        # if torch.isnan(batch_inputs_one_hot).any():
        #     print(f"[NaN DETECTED] Epoch {epoch+1}: Final one-hot input shape: {batch_inputs_one_hot.shape}, min: {batch_inputs_one_hot.min():.6f}, max: {batch_inputs_one_hot.max():.6f}")
        
        optimizer.zero_grad()
        
        # Process the entire batch at once through vocab transformer
        # batch_inputs_one_hot shape: [batch_size, seq_len, vocab_size] - one-hot encodings
        pred_probs = vocab_model(batch_inputs_one_hot, mask=mask)  # [batch_size, seq_len, vocab_size]
        # if torch.isnan(pred_probs).any():
        #     print(f"[NaN DETECTED] Epoch {epoch+1}: Model output pred_probs shape: {pred_probs.shape}, min: {pred_probs.min():.6f}, max: {pred_probs.max():.6f}")
        
        # Extract predictions from the last position
        pred_last = pred_probs[:, -1, :]  # [batch_size, vocab_size]
        # if torch.isnan(pred_last).any():
        #     print(f"[NaN DETECTED] Epoch {epoch+1}: pred_last shape: {pred_last.shape}, min: {pred_last.min():.6f}, max: {pred_last.max():.6f}")
        
        # Convert batch_targets to one-hot vectors for BCE loss
        batch_targets_one_hot = F.one_hot(batch_targets, num_classes=vocab_model.vocab_size).float().to(device)
        # if torch.isnan(batch_targets_one_hot).any():
        #     print(f"[NaN DETECTED] Epoch {epoch+1}: batch_targets_one_hot shape: {batch_targets_one_hot.shape}, min: {batch_targets_one_hot.min():.6f}, max: {batch_targets_one_hot.max():.6f}")
        
        # Use BCE loss with the probability predictions and binary targets
        loss = F.binary_cross_entropy(pred_last, batch_targets_one_hot)
        
        # Add manual L2 regularization to the loss if weight_decay > 0
        if weight_decay > 0:
            l2_reg = torch.tensor(0.0, device=device)
            for param in vocab_model.parameters():
                if param.requires_grad:
                    l2_reg += torch.norm(param) ** 2
            loss += weight_decay * l2_reg
        # if torch.isnan(loss):
        #     print(f"[NaN DETECTED] Epoch {epoch+1}: Loss computed: {loss.item():.6f}")
        
        # Calculate accuracy by comparing predicted classes with target classes
        pred_classes = torch.argmax(pred_last, dim=1)  # [batch_size]
        batch_accuracy = (pred_classes == batch_targets).float().mean()
        # if torch.isnan(batch_accuracy):
        #     print(f"[NaN DETECTED] Epoch {epoch+1}: batch_accuracy: {batch_accuracy.item():.6f}")

        # Comprehensive NaN detection - check all intermediate variables
        # nan_detected = False
        # nan_variable = None
        
        # if torch.isnan(batch_inputs_one_hot).any():
        #     nan_detected = True
        #     nan_variable = "batch_inputs_one_hot"
        # elif torch.isnan(pred_probs).any():
        #     nan_detected = True
        #     nan_variable = "pred_probs (model output)"
        # elif torch.isnan(pred_last).any():
        #     nan_detected = True
        #     nan_variable = "pred_last (last position predictions)"
        # elif torch.isnan(batch_targets_one_hot).any():
        #     nan_detected = True
        #     nan_variable = "batch_targets_one_hot"
        # elif torch.isnan(loss):
        #     nan_detected = True
        #     nan_variable = "loss"
        # elif torch.isnan(batch_accuracy):
        #     nan_detected = True
        #     nan_variable = "batch_accuracy"
        
        # # Check model parameters for NaN
        # if not nan_detected:
        #     for name, param in vocab_model.named_parameters():
        #         if torch.isnan(param).any():
        #             nan_detected = True
        #             nan_variable = f"model parameter: {name}"
        #             break
        
        # if nan_detected:
        #     print(f"\nNaN detected in '{nan_variable}' at epoch {epoch+1}! Stopping training.")
        #     print(f"Last valid loss: {Total_loss[-1] if Total_loss else 'N/A'}")
        #     print(f"Current loss value: {loss.item() if not torch.isnan(loss) else 'NaN'}")
        #     print(f"Current accuracy: {batch_accuracy.item() if not torch.isnan(batch_accuracy) else 'NaN'}")
        #     break
        
        loss.backward()
        
        # Gradient clipping for stability
        # torch.nn.utils.clip_grad_norm_(vocab_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        time_epoch = time.time() - prev_time
        Time.append(time_epoch)
        Total_loss.append(loss.item())
        Accuracy.append(batch_accuracy.item())
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Get some sample predictions for debugging
            # sample_pred_class = pred_classes[0].item() if len(pred_classes) > 0 else 0
            # sample_target = batch_targets[0].item() if len(batch_targets) > 0 else 0
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}, Accuracy: {batch_accuracy.item():.4f}")
            # print(f"  Sample: Predicted class_{sample_pred_class} (target: class_{sample_target})")

    # Save the vocabulary transformer model
    save_transformer_weights(vocab_model, "../model/transformer_final.pth", include_metadata=True)
    print("Final vocabulary transformer saved as 'transformer_final.pth'")
    
    plt.plot(Total_loss[50:])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.show()

    plt.plot(Accuracy[50:])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Over Time")
    plt.show()

    # Save simple arrays for easy extraction later
    num_layers = len(vocab_model.layers)

    if tree_used:
        if seed is not None:
            save_path = f"../data/{num_layers}layer_n{seq_len}_tree_seed{seed}.npz"
        else:
            save_path = f"../data/{num_layers}layer_n{seq_len}_tree.npz"
    else:
        if seed is not None:
            save_path = f"../data/{num_layers}layer_n{seq_len}_seed{seed}.npz"
        else:
            save_path = f"../data/{num_layers}layer_n{seq_len}.npz"

    # Save as NumPy file with just the essential data
    save_data = {
        'timestamps': np.array(Time),
        'accuracy': np.array(Accuracy),
        'loss': np.array(Total_loss),
        'num_epochs': num_epochs,
        'num_layers': num_layers,
        'batches': batch_size,
        'seed': seed if seed is not None else -1  # Save -1 for no seed
    }
    
    np.savez(save_path, **save_data)

    print(f"Training data saved to: {save_path}")
    print(f"Arrays saved: timestamps({len(Time)}), accuracy({len(Accuracy)}), loss({len(Total_loss)})")
    
    print(f"\nVocabulary transformer training completed!")
    print(f"Final accuracy: {Accuracy[-1]:.4f}")
    print(f"Final loss: {Total_loss[-1]:.6f}")


    return vocab_model





def create_transformer_model(config_path=None, device='mps', seed=None):
    """
    Create and return a transformer model
    
    Args:
        config_path: Path to YAML config file. If None, creates a default config.
        device: Device to place the model on ('mps', 'cuda', 'cpu')
        seed: Random seed for model initialization (default: None)
    
    Returns:
        Transformer: The created model
    """
    
    # Set seed for model initialization if provided
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
        torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    
    if config_path is None:
        # Create a default training-optimized config
        config_path = create_sample_config("training_config.yaml")
        print(f"Created default config: {config_path}")
    
    # Load model with device support and seed
    model = Transformer(config_path, device=device, seed=seed)
    print(f"Model created from config: {config_path} on device: {device}")
    if seed is not None:
        print(f"Model initialized with seed: {seed}")
    
    return model

def save_transformer_weights(model, filepath="transformer_weights.pth", include_metadata=True):
    """
    Save transformer model weights and metadata to a file
    
    Args:
        model: The transformer model to save
        filepath: Path where to save the weights (default: "transformer_weights.pth")
        include_metadata: Whether to include model metadata (default: True)
    """
    save_data = {
        'model_state_dict': model.state_dict(),
    }
    
    if include_metadata:
        # Add metadata about the model
        save_data.update({
            'model_config': model.config if hasattr(model, 'config') else None,
            'num_layers': len(model.layers) if hasattr(model, 'layers') else None,
            'model_class': model.__class__.__name__,
            'save_timestamp': torch.tensor([1.0], dtype=torch.float32)  # Simple timestamp placeholder
        })
    
    torch.save(save_data, filepath)
    print(f"Transformer weights saved to: {filepath}")
    
    # Print model size info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Saved model with {total_params:,} total parameters ({trainable_params:,} trainable)")

def load_transformer_weights(model, filepath="transformer_weights.pth", strict=True):
    """
    Load transformer model weights from a file
    
    Args:
        model: The transformer model to load weights into
        filepath: Path to the saved weights file
        strict: Whether to strictly enforce that the keys match (default: True)
    
    Returns:
        dict: Metadata from the saved file (if available)
    """
    # Load the saved data
    checkpoint = torch.load(filepath, map_location='cpu')  # Load to CPU first
    
    # Extract model state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    else:
        # Assume it's just the state dict
        state_dict = checkpoint
        metadata = {}
    
    # Load weights into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    print(f"Transformer weights loaded from: {filepath}")
    
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys}")
    
    # Print metadata if available
    if metadata:
        print("Loaded model metadata:")
        for key, value in metadata.items():
            if key != 'save_timestamp':
                print(f"  {key}: {value}")
    
    return metadata

def create_model_from_weights(weights_filepath, config_path=None, device='mps'):
    """
    Create a new transformer model and load weights from file
    
    Args:
        weights_filepath: Path to the saved weights file
        config_path: Path to config file (if None, tries to use metadata or default)
        device: Device to place the model on ('mps', 'cuda', 'cpu')
    
    Returns:
        Loaded transformer model
    """
    # Try to load metadata first to get config info
    checkpoint = torch.load(weights_filepath, map_location='cpu')
    saved_config = None
    
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        saved_config = checkpoint['model_config']
        if saved_config:
            print(f"Found saved model configuration: {saved_config}")
            # Create a temporary config file from saved metadata
            import tempfile
            import yaml
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(saved_config, f)
                temp_config_path = f.name
            print(f"Using config from saved metadata")
            config_path = temp_config_path
    
    if config_path is None:
        print("No config found in saved model and no config_path provided. Creating default config.")
        config_path = create_sample_config("temp_config.yaml")
    
    # Create model
    model = create_transformer_model(config_path, device=device)
    
    # Load weights
    metadata = load_transformer_weights(model, weights_filepath, strict=False)
    
    # Clean up temporary file if we created one
    if saved_config and 'temp_config_path' in locals():
        import os
        try:
            os.unlink(temp_config_path)
        except:
            pass
    
    return model

if __name__ == "__main__":
    # Load configuration from YAML file
    config_file = "../model/training_config.yaml"
    
    
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract training parameters from config
    training_config = config.get('training', {})
    testing_config = config.get('testing', {})
    
    print(f"Loaded configuration from {config_file}")
    print(f"Training parameters: {training_config}")
    print(f"Testing parameters: {testing_config}")
    
    # Get seed from config
    seed = training_config.get('seed', None)
    if seed is not None:
        print(f"Using seed from config: {seed}")
        # Set global seed at the very beginning using our utility function
        set_global_seed(seed)
    
    # Setup device - prefer config, then MPS if available, fallback to CPU
    device_name = training_config.get('device', 'mps')
    device = setup_device(device_name)
    
    # Validate required parameters
    required_training_params = ['seq_len', 'num_epochs', 'learning_rate', 'batch_size']
    for param in required_training_params:
        if param not in training_config:
            raise ValueError(f"Missing required training parameter: {param}")
    
    
    # Create the transformer model
    print("\nCreating transformer model...")
    transformer_model = create_transformer_model(config_file, device=device, seed=seed)
    
    # Train the transformer using parameters from config
    print("\nStarting training...")
    # trained_model = train_transformer_from_dataset(
    #     model=transformer_model,
    #     vectors_file=training_config['vectors_file'],
    #     seq_len=training_config['seq_len'],
    #     d_model=training_config['d_model'],
    #     num_epochs=training_config['num_epochs'],
    #     learning_rate=training_config['learning_rate'],
    #     batch_size=training_config['batch_size']
    # )
    fold = 2
    trained_vocab_model = train_transformer(
        model=transformer_model,
        device=device,
        seq_len=training_config['seq_len'],
        num_epochs=training_config['num_epochs'],
        learning_rate=training_config['learning_rate'],
        batch_size=training_config['batch_size'],
        n=training_config['seq_len']//fold,
        x=None,  # Example input, can be adjusted
        fold=fold,
        weight_decay=training_config.get('weight_decay', 0.0),
        seed=61
    )

    
    # Demonstrate saving and loading weights
    print("\n" + "="*50)
    print("DEMONSTRATING SAVE/LOAD FUNCTIONALITY")
    print("="*50)
    
    
    print("Training completed successfully!")
    print(f"Trained vocabulary transformer model: {type(trained_vocab_model).__name__}")
    print(f"Model saved to: ../model/transformer_final.pth")
