import jax
import jax.numpy as jnp
import flax.linen as nn

class Diagonal3TimeResLayer(nn.Module):
    hidden_size: int
    is_last: bool

    def setup(self):
        self.diag_main = self.param('diag_main', jax.nn.initializers.zeros, (self.hidden_size,))
        self.diag_up = self.param('diag_up', jax.nn.initializers.zeros, (self.hidden_size - 1,))
        self.diag_down = self.param('diag_down', jax.nn.initializers.zeros, (self.hidden_size - 1,))
        self.bias = self.param('bias', jax.nn.initializers.zeros, (self.hidden_size,))

        if not self.is_last:
            self.w_down = nn.Dense(
                self.hidden_size, 
                kernel_init=nn.initializers.variance_scaling(2.0, "fan_in", "uniform"), 
                use_bias=False
            )

    def __call__(self, x_proj_t, h_prev, h_next_layer_prev=None):
        main_term = h_prev * self.diag_main

        up_val = h_prev[:, 1:] * self.diag_up
        up_term = jnp.concatenate([up_val, jnp.zeros((h_prev.shape[0], 1))], axis=1)

        down_val = h_prev[:, :-1] * self.diag_down
        down_term = jnp.concatenate([jnp.zeros((h_prev.shape[0], 1)), down_val], axis=1)

        res_time = main_term + up_term + down_term

        res = x_proj_t + res_time + self.bias
        if not self.is_last and h_next_layer_prev is not None:
            res = res + self.w_down(h_next_layer_prev)

        delta = jax.nn.leaky_relu(res, negative_slope=0.1)
        return h_prev + delta


class Diagonal3TimeResModel(nn.Module):
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int = 1

    def setup(self):
        self.w_up_layers = [
            nn.Dense(
                self.hidden_size, 
                kernel_init=nn.initializers.variance_scaling(2.0, "fan_in", "uniform"), 
                use_bias=False
            )
            for _ in range(self.num_layers)
        ]
        
        self.rec_layers = [
            Diagonal3TimeResLayer(hidden_size=self.hidden_size, is_last=(i == self.num_layers - 1))
            for i in range(self.num_layers)
        ]
        
        self.fc = nn.Dense(self.output_size)

    def __call__(self, x):
        batch_size, seq_len, _ = x.shape

        # Предварительный расчет проекций
        x_projected_list = [w_up(x) for w_up in self.w_up_layers]
        x_projected = jnp.stack(x_projected_list, axis=0)  # (num_layers, batch_size, seq_len, hidden_size)

        # ФОРСИРОВАННАЯ ИНИЦИАЛИЗАЦИЯ: вызываем слои один раз фиктивно вне lax.scan
        if self.is_initializing():
            for l in range(self.num_layers):
                dummy_x = jnp.zeros((batch_size, self.hidden_size))
                dummy_h = jnp.zeros((batch_size, self.hidden_size))
                h_next = dummy_h if l + 1 < self.num_layers else None
                _ = self.rec_layers[l](dummy_x, dummy_h, h_next)

        # Транспонируем для сканирования
        x_projected_t = jnp.transpose(x_projected, (2, 0, 1, 3))  # (seq_len, num_layers, batch_size, hidden_size)

        init_h = jnp.zeros((self.num_layers, batch_size, self.hidden_size))

        def scan_fn(carry_h, x_t):
            new_h = []
            for l in range(self.num_layers):
                h_prev = carry_h[l]
                h_next = carry_h[l+1] if l + 1 < self.num_layers else None
                h_curr_new = self.rec_layers[l](x_t[l], h_prev, h_next)
                new_h.append(h_curr_new)
            
            new_h_stacked = jnp.stack(new_h, axis=0)
            return new_h_stacked, None

        final_h, _ = jax.lax.scan(scan_fn, init_h, x_projected_t)

        return self.fc(final_h[-1])