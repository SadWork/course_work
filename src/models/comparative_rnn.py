# Путь: src/models/comparative_rnn.py

import math
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

class ComparativeRNNLayer(nn.Module):
    hidden_size: int
    spatial_dim: int
    layer_idx: int
    num_layers: int  # L
    seq_len: int     # T
    model_type: str = 'standard'  # 'standard', 'spatial', 'temporal', 'alternating', 'random_balanced', 'random_uniform'
    seed: int = 42

    def setup(self):
        if self.model_type in ['random_balanced', 'random_uniform']:
            rng = np.random.default_rng(self.seed)
            
            if self.model_type == 'random_uniform':
                indices_np = rng.choice(self.hidden_size + self.spatial_dim, size=self.hidden_size, replace=True)
            else:
                source_channels = rng.choice([0, 1], size=self.hidden_size, replace=True, p=[0.5, 0.5])
                indices_np = np.zeros(self.hidden_size, dtype=np.int32)
                
                for i in range(self.hidden_size):
                    if source_channels[i] == 0:
                        indices_np[i] = rng.choice(self.hidden_size)
                    else:
                        indices_np[i] = self.hidden_size + rng.choice(self.spatial_dim)
                        
            self.indices = jnp.array(indices_np, dtype=jnp.int32)

    @nn.compact
    def __call__(self, h_spatial_prev, h_temporal_prev):
        batch_size, spatial_dim = h_spatial_prev.shape
        
        # Определяем масштаб начальной дисперсии весов рекуррентного блока
        if self.model_type == 'spatial':
            scale_val = 1.0 / self.num_layers
        elif self.model_type == 'temporal':
            scale_val = 1.0 / np.sqrt(self.seq_len)
        elif self.model_type == 'alternating':
            if self.layer_idx % 2 == 0:
                scale_val = 1.0 / self.seq_len
            else:
                scale_val = 1.0 / self.num_layers
        elif self.model_type == 'random_uniform':
            scale_val = min(1.0 / np.sqrt(self.seq_len), 1.0 / np.sqrt(self.num_layers))
        else:
            scale_val = 1.0
            
        w_up = nn.Dense(
            self.hidden_size, 
            use_bias=False, 
            name="w_up",
            # Масштабируем дисперсию инициализации весов
            kernel_init=jax.nn.initializers.variance_scaling(scale=scale_val, mode="fan_in", distribution="normal")
        )
        w_rec = nn.Dense(
            self.hidden_size, 
            use_bias=True, 
            name="w_rec",
            kernel_init=jax.nn.initializers.variance_scaling(scale=scale_val, mode="fan_in", distribution="normal")
        )
        
        cell_input = w_up(h_spatial_prev) + w_rec(h_temporal_prev)
        delta = jax.nn.tanh(cell_input)
        
        # Остаточные связи строго равны 1.0, чтобы градиенты не затухали
        if self.model_type == 'standard':
            return delta
            
        elif self.model_type == 'spatial':
            if spatial_dim != self.hidden_size:
                return delta
            else:
                return delta + h_spatial_prev
            
        elif self.model_type == 'temporal':
            return delta + h_temporal_prev
            
        elif self.model_type == 'alternating':
            if self.layer_idx % 2 == 0:
                return delta + h_temporal_prev
            else:
                if spatial_dim != self.hidden_size:
                    return delta
                else:
                    return delta + h_spatial_prev
                
        elif self.model_type in ['random_balanced', 'random_uniform']:
            concat_state = jnp.concatenate([h_temporal_prev, h_spatial_prev], axis=-1)
            skip = concat_state[:, self.indices]
            return delta + skip
            
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")


class ComparativeStackedRNN(nn.Module):
    input_size: int
    hidden_size: int
    num_layers: int
    seq_len: int
    output_size: int = 10
    model_type: str = 'standard'
    seed: int = 42

    def setup(self):
        self.layers = [
            ComparativeRNNLayer(
                hidden_size=self.hidden_size,
                spatial_dim=self.input_size if l == 0 else self.hidden_size,
                layer_idx=l,
                num_layers=self.num_layers,
                seq_len=self.seq_len,
                model_type=self.model_type,
                seed=self.seed + l
            )
            for l in range(self.num_layers)
        ]
        self.fc = nn.Dense(self.output_size)

    def __call__(self, x):
        batch_size, seq_len, input_size = x.shape

        if self.is_initializing():
            current_input_dim = input_size
            for l in range(self.num_layers):
                dummy_spatial = jnp.zeros((batch_size, current_input_dim))
                dummy_temporal = jnp.zeros((batch_size, self.hidden_size))
                _ = self.layers[l](dummy_spatial, dummy_temporal)
                current_input_dim = self.hidden_size

        x_t = jnp.transpose(x, (1, 0, 2))
        init_h = jnp.zeros((self.num_layers, batch_size, self.hidden_size))

        def scan_fn(carry_h, x_step):
            new_h = []
            curr_spatial = x_step
            for l in range(self.num_layers):
                h_temporal_prev = carry_h[l]
                h_curr_new = self.layers[l](curr_spatial, h_temporal_prev)
                new_h.append(h_curr_new)
                curr_spatial = h_curr_new
            new_h_stacked = jnp.stack(new_h, axis=0)
            return new_h_stacked, None

        final_h, _ = jax.lax.scan(scan_fn, init_h, x_t)
        return self.fc(final_h[-1])