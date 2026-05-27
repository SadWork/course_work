# Путь: src/models/sparse_random_time_res.py

import math
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

def generate_random_connections(hidden_size, k, seed=42):
    rng = np.random.default_rng(seed)
    indices = []
    for i in range(hidden_size):
        row = [i]
        possible_targets = [j for j in range(hidden_size) if j != i]
        if len(possible_targets) >= k:
            chosen = rng.choice(possible_targets, size=k, replace=False).tolist()
        else:
            chosen = rng.choice(range(hidden_size), size=k, replace=True).tolist()
        row.extend(chosen)
        indices.append(row)
    return jnp.array(indices, dtype=jnp.int32)


class SparseRandomTimeResLayer(nn.Module):
    hidden_size: int
    is_last: bool
    is_first: bool
    k: int
    connection_seed: int = 42

    def setup(self):
        self.indices = generate_random_connections(
            self.hidden_size, self.k, self.connection_seed
        )

    @nn.compact
    def __call__(self, h_spatial_prev, h_temporal_prev, h_next_layer_prev=None):
        w_up = nn.Dense(self.hidden_size, use_bias=False, name="w_up")
        up_term = w_up(h_spatial_prev)

        weights = self.param(
            'weights', 
            jax.nn.initializers.zeros, 
            (self.hidden_size, self.k + 1)
        )
        h_gathered = jnp.take(h_temporal_prev, self.indices, axis=1)
        time_term = jnp.sum(h_gathered * weights, axis=-1)

        res = up_term + time_term + self.param('bias', jax.nn.initializers.zeros, (self.hidden_size,))

        if not self.is_last and h_next_layer_prev is not None:
            w_down = nn.Dense(self.hidden_size, use_bias=False, name="w_down")
            res = res + w_down(h_next_layer_prev)

        delta = jax.nn.leaky_relu(res, negative_slope=0.1)

        if not self.is_first and h_spatial_prev.shape[-1] == self.hidden_size:
            return delta + h_spatial_prev
        else:
            return delta


class SparseRandomTimeResModel(nn.Module):
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int = 1
    target_diameter: float = None  
    k: int = None                  
    connection_seed: int = 42

    def setup(self):
        if self.k is not None:
            self.active_k = self.k
        elif self.target_diameter is not None:
            calculated_k = math.ceil(self.hidden_size ** (1.0 / self.target_diameter))
            self.active_k = max(1, min(calculated_k, self.hidden_size - 1))
        else:
            self.active_k = 2  

        self.rec_layers = [
            SparseRandomTimeResLayer(
                hidden_size=self.hidden_size, 
                is_last=(i == self.num_layers - 1),
                is_first=(i == 0),
                k=self.active_k,
                connection_seed=self.connection_seed + i
            )
            for i in range(self.num_layers)
        ]
        self.fc = nn.Dense(self.output_size)

    def __call__(self, x):
        batch_size, seq_len, input_size = x.shape

        if self.is_initializing():
            current_input_dim = input_size
            for l in range(self.num_layers):
                dummy_spatial = jnp.zeros((batch_size, current_input_dim))
                dummy_temporal = jnp.zeros((batch_size, self.hidden_size))
                h_next = dummy_temporal if l + 1 < self.num_layers else None
                _ = self.rec_layers[l](dummy_spatial, dummy_temporal, h_next)
                current_input_dim = self.hidden_size

        x_t = jnp.transpose(x, (1, 0, 2))  
        init_h = jnp.zeros((self.num_layers, batch_size, self.hidden_size))

        def scan_fn(carry_h, x_step):
            new_h = []
            curr_spatial = x_step  
            
            for l in range(self.num_layers):
                h_temporal_prev = carry_h[l]
                h_next_layer_prev = carry_h[l+1] if l + 1 < self.num_layers else None
                
                h_curr_new = self.rec_layers[l](curr_spatial, h_temporal_prev, h_next_layer_prev)
                new_h.append(h_curr_new)
                curr_spatial = h_curr_new  
                
            new_h_stacked = jnp.stack(new_h, axis=0)
            return new_h_stacked, None

        final_h, _ = jax.lax.scan(scan_fn, init_h, x_t)
        return self.fc(final_h[-1])

    def get_cyclic_graph(self, hidden_size):
        """Возвращает точный циклический граф структуры модели."""
        import math
        if self.k is not None:
            active_k = self.k
        elif self.target_diameter is not None:
            calculated_k = math.ceil(hidden_size ** (1.0 / self.target_diameter))
            active_k = max(1, min(calculated_k, hidden_size - 1))
        else:
            active_k = 2

        edges = []
        for i in range(hidden_size):
            edges.append(('X', ('H', 0, i), 0))
            
        for l in range(self.num_layers - 1):
            for i in range(hidden_size):
                for j in range(hidden_size):
                    edges.append((('H', l, i), ('H', l + 1, j), 0))
                
        for i in range(hidden_size):
            edges.append((('H', self.num_layers - 1, i), 'Y', 0))
            
        for l in range(self.num_layers):
            indices = generate_random_connections(hidden_size, active_k, self.connection_seed + l)
            for i in range(hidden_size):
                for src in indices[i]:
                    edges.append((('H', l, int(src)), ('H', l, i), 1))
                    
        for l in range(self.num_layers - 1):
            for i in range(hidden_size):
                for j in range(hidden_size):
                    edges.append((('H', l + 1, j), ('H', l, i), 1))
        return edges

    def get_computation_graph(self):
        """Динамически генерирует вычислительный граф со случайной разреженностью (мета-описание)."""
        import math
        if self.k is not None:
            active_k = self.k
        elif self.target_diameter is not None:
            calculated_k = math.ceil(self.hidden_size ** (1.0 / self.target_diameter))
            active_k = max(1, min(calculated_k, self.hidden_size - 1))
        else:
            active_k = 2

        nodes = []
        edges = []
        
        nodes.append({'id': 'X', 'label': 'X', 'type': 'input'})
        
        for l in range(self.num_layers):
            is_first = (l == 0)
            is_last = (l == self.num_layers - 1)
            
            nodes.append({'id': f'Dense_Up_{l}', 'label': '$W_{up}$\n(Dense)', 'type': 'op'})
            nodes.append({'id': f'Sparse_Rec_{l}', 'label': f'Sparse Rec\n(k={active_k})', 'type': 'op'})
            
            if not is_last:
                nodes.append({'id': f'Dense_Down_{l}', 'label': '$W_{down}$\n(Dense)', 'type': 'op'})
                
            nodes.append({'id': f'Sum_{l}', 'label': '+', 'type': 'sum'})
            nodes.append({'id': f'Act_{l}', 'label': 'LeakyReLU\n(0.1)', 'type': 'activation'})
            
            if not is_first:
                nodes.append({'id': f'ResSum_{l}', 'label': '+ (Residual)', 'type': 'sum'})
                
            nodes.append({'id': f'H_{l}', 'label': f'H_{l}', 'type': 'state'})
            
            prev_state = 'X' if l == 0 else f'H_{l-1}'
            edges.append((prev_state, f'Dense_Up_{l}', 0))
            edges.append((f'H_{l}', f'Sparse_Rec_{l}', 1))
            
            if not is_last:
                edges.append((f'H_{l+1}', f'Dense_Down_{l}', 1))
                
            edges.append((f'Dense_Up_{l}', f'Sum_{l}', 0))
            edges.append((f'Sparse_Rec_{l}', f'Sum_{l}', 0))
            if not is_last:
                edges.append((f'Dense_Down_{l}', f'Sum_{l}', 0))
                
            edges.append((f'Sum_{l}', f'Act_{l}', 0))
            
            if not is_first:
                edges.append((f'Act_{l}', f'ResSum_{l}', 0))
                edges.append((prev_state, f'ResSum_{l}', 0))
                edges.append((f'ResSum_{l}', f'H_{l}', 0))
            else:
                edges.append((f'Act_{l}', f'H_{l}', 0))
                
        # Классификатор
        nodes.append({'id': 'FC', 'label': 'FC\n(Dense)', 'type': 'op'})
        nodes.append({'id': 'Y', 'label': 'Y', 'type': 'output'})
        
        edges.append((f'H_{self.num_layers-1}', 'FC', 0))
        edges.append(('FC', 'Y', 0))
        
        return nodes, edges