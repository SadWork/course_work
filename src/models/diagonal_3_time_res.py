# Путь: src/models/diagonal_3_time_res.py

import jax
import jax.numpy as jnp
import flax.linen as nn

class Diagonal3TimeResLayer(nn.Module):
    hidden_size: int
    is_last: bool
    is_first: bool

    @nn.compact
    def __call__(self, h_spatial_prev, h_temporal_prev, h_next_layer_prev=None):
        w_up = nn.Dense(
            self.hidden_size, 
            use_bias=False, 
            name="w_up",
            kernel_init=jax.nn.initializers.variance_scaling(scale=0.001, mode="fan_in", distribution="normal")
        )
        up_term = w_up(h_spatial_prev)

        diag_main = self.param('diag_main', jax.nn.initializers.zeros, (self.hidden_size,))
        diag_up = self.param('diag_up', jax.nn.initializers.zeros, (self.hidden_size - 1,))
        diag_down = self.param('diag_down', jax.nn.initializers.zeros, (self.hidden_size - 1,))

        main_term = h_temporal_prev * diag_main

        up_val = h_temporal_prev[:, 1:] * diag_up
        up_term_t = jnp.concatenate([up_val, jnp.zeros((h_temporal_prev.shape[0], 1))], axis=1)

        down_val = h_temporal_prev[:, :-1] * diag_down
        down_term_t = jnp.concatenate([jnp.zeros((h_temporal_prev.shape[0], 1)), down_val], axis=1)

        time_term = main_term + up_term_t + down_term_t
        res = up_term + time_term + self.param('bias', jax.nn.initializers.zeros, (self.hidden_size,))

        if not self.is_last and h_next_layer_prev is not None:
            w_down = nn.Dense(
                self.hidden_size, 
                use_bias=False,
                name="w_down",
                kernel_init=jax.nn.initializers.variance_scaling(scale=0.001, mode="fan_in", distribution="normal")
            )
            res = res + w_down(h_next_layer_prev)

        delta = jax.nn.leaky_relu(res, negative_slope=0.1)

        # Изменение: Остаточная связь идет по времени (добавляем h_temporal_prev вместо h_spatial_prev)
        return delta + h_temporal_prev


class Diagonal3TimeResModel(nn.Module):
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int = 1

    def setup(self):
        self.rec_layers = [
            Diagonal3TimeResLayer(
                hidden_size=self.hidden_size, 
                is_last=(i == self.num_layers - 1),
                is_first=(i == 0)
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
            for i in range(hidden_size):
                edges.append((('H', l, i), ('H', l, i), 1))
                if i + 1 < hidden_size:
                    edges.append((('H', l, i), ('H', l, i + 1), 1))
                if i - 1 >= 0:
                    edges.append((('H', l, i), ('H', l, i - 1), 1))
                    
        for l in range(self.num_layers - 1):
            for i in range(hidden_size):
                for j in range(hidden_size):
                    edges.append((('H', l + 1, j), ('H', l, i), 1))
        return edges

    def get_computation_graph(self):
        """Динамически генерирует мета-описание вычислительного графа с временными остаточными связями."""
        nodes = []
        edges = []
        
        nodes.append({'id': 'X', 'label': 'X', 'type': 'input'})
        
        for l in range(self.num_layers):
            is_last = (l == self.num_layers - 1)
            
            nodes.append({'id': f'Dense_Up_{l}', 'label': '$W_{up}$\n(Dense)', 'type': 'op'})
            nodes.append({'id': f'Tridiag_{l}', 'label': 'Tridiagonal\nRecurrence', 'type': 'op'})
            
            if not is_last:
                nodes.append({'id': f'Dense_Down_{l}', 'label': '$W_{down}$\n(Dense)', 'type': 'op'})
                
            nodes.append({'id': f'Sum_{l}', 'label': '+', 'type': 'sum'})
            nodes.append({'id': f'Act_{l}', 'label': 'LeakyReLU\n(0.1)', 'type': 'activation'})
            
            # Остаточный сумматор теперь присутствует на каждом слое для суммирования во времени
            nodes.append({'id': f'ResSum_{l}', 'label': '+ (Temp Res)', 'type': 'sum'})
            nodes.append({'id': f'H_{l}', 'label': f'H_{l}', 'type': 'state'})
            
            # Связи
            prev_state = 'X' if l == 0 else f'H_{l-1}'
            edges.append((prev_state, f'Dense_Up_{l}', 0))
            edges.append((f'H_{l}', f'Tridiag_{l}', 1))
            
            if not is_last:
                edges.append((f'H_{l+1}', f'Dense_Down_{l}', 1))
                
            edges.append((f'Dense_Up_{l}', f'Sum_{l}', 0))
            edges.append((f'Tridiag_{l}', f'Sum_{l}', 0))
            if not is_last:
                edges.append((f'Dense_Down_{l}', f'Sum_{l}', 0))
                
            edges.append((f'Sum_{l}', f'Act_{l}', 0))
            
            # Временная остаточная связь (LeakyReLU(x) + H_{l, t-1})
            edges.append((f'Act_{l}', f'ResSum_{l}', 0))
            edges.append((f'H_{l}', f'ResSum_{l}', 1))     # Связь по времени (sigma=1)
            edges.append((f'ResSum_{l}', f'H_{l}', 0))
                
        # Выходной классификатор
        nodes.append({'id': 'FC', 'label': 'FC\n(Dense)', 'type': 'op'})
        nodes.append({'id': 'Y', 'label': 'Y', 'type': 'output'})
        
        edges.append((f'H_{self.num_layers-1}', 'FC', 0))
        edges.append(('FC', 'Y', 0))
        
        return nodes, edges
