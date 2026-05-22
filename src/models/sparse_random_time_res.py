import math
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

def generate_random_connections(hidden_size, k, seed=42):
    """
    Генерирует статическую матрицу индексов связей.
    Каждая строка i содержит индекс i (self-connection) и k уникальных случайных индексов.
    """
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
    k: int
    connection_seed: int = 42

    def setup(self):
        # Статические индексы связей (не являются параметрами обучения)
        self.indices = generate_random_connections(
            self.hidden_size, self.k, self.connection_seed
        )
        
        # Обучаемые веса переходов для каждой из (k + 1) входящих связей
        self.weights = self.param(
            'weights', 
            jax.nn.initializers.zeros, 
            (self.hidden_size, self.k + 1)
        )
        self.bias = self.param('bias', jax.nn.initializers.zeros, (self.hidden_size,))

        if not self.is_last:
            self.w_down = nn.Dense(
                self.hidden_size, 
                kernel_init=nn.initializers.variance_scaling(2.0, "fan_in", "uniform"), 
                use_bias=False
            )

    def __call__(self, x_proj_t, h_prev, h_next_layer_prev=None):
        # Собираем состояния связанных нейронов: (batch_size, hidden_size, k + 1)
        h_gathered = jnp.take(h_prev, self.indices, axis=1)

        # Вычисляем взвешенную сумму по связям
        res_time = jnp.sum(h_gathered * self.weights, axis=-1)

        res = x_proj_t + res_time + self.bias
        if not self.is_last and h_next_layer_prev is not None:
            res = res + self.w_down(h_next_layer_prev)

        delta = jax.nn.leaky_relu(res, negative_slope=0.1)
        return h_prev + delta


class SparseRandomTimeResModel(nn.Module):
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int = 1
    target_diameter: float = None  # Желаемый диаметр графа
    k: int = None                  # Альтернативно можно задать k напрямую
    connection_seed: int = 42

    def setup(self):
        # Определение k на основе желаемого диаметра
        if self.k is not None:
            active_k = self.k
        elif self.target_diameter is not None:
            calculated_k = math.ceil(self.hidden_size ** (1.0 / self.target_diameter))
            active_k = max(1, min(calculated_k, self.hidden_size - 1))
        else:
            active_k = 2  # Значение по умолчанию

        self.w_up_layers = [
            nn.Dense(
                self.hidden_size, 
                kernel_init=nn.initializers.variance_scaling(2.0, "fan_in", "uniform"), 
                use_bias=False
            )
            for _ in range(self.num_layers)
        ]
        
        self.rec_layers = [
            SparseRandomTimeResLayer(
                hidden_size=self.hidden_size, 
                is_last=(i == self.num_layers - 1),
                k=active_k,
                connection_seed=self.connection_seed + i
            )
            for i in range(self.num_layers)
        ]
        
        self.fc = nn.Dense(self.output_size)

    def __call__(self, x):
        batch_size, seq_len, _ = x.shape

        x_projected_list = [w_up(x) for w_up in self.w_up_layers]
        x_projected = jnp.stack(x_projected_list, axis=0)  

        # Форсированная инициализация вне lax.scan
        if self.is_initializing():
            for l in range(self.num_layers):
                dummy_x = jnp.zeros((batch_size, self.hidden_size))
                dummy_h = jnp.zeros((batch_size, self.hidden_size))
                h_next = dummy_h if l + 1 < self.num_layers else None
                _ = self.rec_layers[l](dummy_x, dummy_h, h_next)

        x_projected_t = jnp.transpose(x_projected, (2, 0, 1, 3))  

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