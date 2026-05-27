# Путь: src/models/standard_rnn.py

import jax
import jax.numpy as jnp
import flax.linen as nn

class StandardRNNCell(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, h, x):
        h_proj = nn.Dense(self.hidden_size, use_bias=False)(h)
        x_proj = nn.Dense(self.hidden_size)(x)
        return jax.nn.tanh(x_proj + h_proj)


class StandardStackedRNN(nn.Module):
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int = 1

    def setup(self):
        self.cells = [StandardRNNCell(hidden_size=self.hidden_size) for _ in range(self.num_layers)]
        self.fc = nn.Dense(self.output_size)

    def __call__(self, x):
        batch_size, seq_len, input_size = x.shape

        if self.is_initializing():
            current_input_dim = input_size  
            for l in range(self.num_layers):
                dummy_x = jnp.zeros((batch_size, current_input_dim))
                dummy_h = jnp.zeros((batch_size, self.hidden_size))
                _ = self.cells[l](dummy_h, dummy_x)
                current_input_dim = self.hidden_size  

        x_t = jnp.transpose(x, (1, 0, 2))  
        init_states = [jnp.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]

        def scan_fn(carry_states, x_step):
            new_states = []
            curr_input = x_step
            for l in range(self.num_layers):
                new_state = self.cells[l](carry_states[l], curr_input)
                new_states.append(new_state)
                curr_input = new_state
            return new_states, None

        final_states, _ = jax.lax.scan(scan_fn, init_states, x_t)
        return self.fc(final_states[-1])

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
                for j in range(hidden_size):
                    edges.append((('H', l, i), ('H', l, j), 1))
        return edges

    def get_computation_graph(self):
        """Динамически генерирует детальный вычислительный граф с учетом количества слоев."""
        nodes = []
        edges = []
        
        # Входной узел
        nodes.append({'id': 'X', 'label': 'X', 'type': 'input', 'pos': (0.0, 0.0)})
        
        for l in range(self.num_layers):
            base_x = (l + 1) * 4.0
            
            # Узлы операций внутри ячейки
            nodes.append({'id': f'Dense_X_{l}', 'label': f'$W_x$\n(Dense)', 'type': 'op', 'pos': (base_x, 1.0)})
            nodes.append({'id': f'Dense_H_{l}', 'label': f'$W_h$\n(Dense)', 'type': 'op', 'pos': (base_x, -1.0)})
            nodes.append({'id': f'Sum_{l}', 'label': '+', 'type': 'sum', 'pos': (base_x + 1.2, 0.0)})
            nodes.append({'id': f'Act_{l}', 'label': 'Tanh', 'type': 'activation', 'pos': (base_x + 2.2, 0.0)})
            nodes.append({'id': f'H_{l}', 'label': f'H_{l}', 'type': 'state', 'pos': (base_x + 3.2, 0.0)})
            
            # Направление прямого пространственного сигнала
            prev_state = 'X' if l == 0 else f'H_{l-1}'
            edges.append((prev_state, f'Dense_X_{l}', 0))
            
            # Временной рекуррентный цикл к Dense_H
            edges.append((f'H_{l}', f'Dense_H_{l}', 1))
            
            # Сложение результатов проекций
            edges.append((f'Dense_X_{l}', f'Sum_{l}', 0))
            edges.append((f'Dense_H_{l}', f'Sum_{l}', 0))
            
            # Применение нелинейности Tanh
            edges.append((f'Sum_{l}', f'Act_{l}', 0))
            edges.append((f'Act_{l}', f'H_{l}', 0))
            
        # Классификатор (FC)
        fc_x = (self.num_layers + 1) * 4.0
        nodes.append({'id': 'FC', 'label': 'FC\n(Dense)', 'type': 'op', 'pos': (fc_x, 0.0)})
        nodes.append({'id': 'Y', 'label': 'Y', 'type': 'output', 'pos': (fc_x + 1.2, 0.0)})
        
        edges.append((f'H_{self.num_layers-1}', 'FC', 0))
        edges.append(('FC', 'Y', 0))
        
        return nodes, edges