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

        # ФОРСИРОВАННАЯ ИНИЦИАЛИЗАЦИЯ: вызываем ячейки с корректными размерностями
        if self.is_initializing():
            current_input_dim = input_size  # Первый слой принимает входной вектор (размерность 2)
            for l in range(self.num_layers):
                dummy_x = jnp.zeros((batch_size, current_input_dim))
                dummy_h = jnp.zeros((batch_size, self.hidden_size))
                _ = self.cells[l](dummy_h, dummy_x)
                current_input_dim = self.hidden_size  # Последующие слои принимают состояния предыдущих (размерность 60)

        x_t = jnp.transpose(x, (1, 0, 2))  # (seq_len, batch_size, input_size)
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