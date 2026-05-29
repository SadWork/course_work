# Путь: src/models/residual_comparison.py

import math
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

class ComparativeResBlock(nn.Module):
    features: int
    block_type: str = 'standard'  # 'standard', 'random_skip', 'no_skip'
    seed: int = 42
    scale: float = 1.0  # Коэффициент масштабирования инициализации весов

    def setup(self):
        if self.block_type == 'random_skip':
            # Матрица P: ровно одна единица в строке, позиции случайны и независимы
            rng = np.random.default_rng(self.seed)
            self.indices = jnp.array(rng.choice(self.features, size=self.features, replace=True))
        else:
            self.indices = None

    @nn.compact
    def __call__(self, x):
        # Масштабируем начальную дисперсию весов, чтобы ограничить рост значений на выходе блока
        kernel_init = jax.nn.initializers.variance_scaling(
            scale=self.scale, 
            mode="fan_in", 
            distribution="normal"
        )
        
        h = nn.Dense(self.features, kernel_init=kernel_init)(x)
        activated = jax.nn.relu(h)
        
        if self.block_type == 'standard':
            # y = x + \sigma(Wx + b)
            return x + activated
        elif self.block_type == 'random_skip':
            # y = Px + \sigma(Wx + b)
            return x[..., self.indices] + activated
        elif self.block_type == 'no_skip':
            # y = \sigma(Wx + b) (обычный глубокий MLP)
            return activated
        else:
            raise ValueError(f"Unknown block type: {self.block_type}")


class ComparativeResNetMLP(nn.Module):
    hidden_size: int
    num_blocks: int
    output_size: int = 10
    block_type: str = 'standard'
    seed: int = 42
    custom_scale: float = None  # Поддержка оптимизированного масштаба

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        
        # Входная проекция
        x = nn.Dense(self.hidden_size)(x)
        x = jax.nn.relu(x)
        
        # Рассчитываем масштаб весов в зависимости от общего числа блоков N.
        # Для no_skip используем стандартный масштаб 1.0, 
        # так как там нет суммирования с обходным путем.
        if self.custom_scale is not None:
            block_scale = self.custom_scale
        elif self.block_type in ['standard', 'random_skip']:
            block_scale = 1.0 / (self.num_blocks)
        else:
            block_scale = 1.0
        
        for i in range(self.num_blocks):
            x = ComparativeResBlock(
                features=self.hidden_size, 
                block_type=self.block_type, 
                seed=self.seed + i,
                scale=block_scale
            )(x)
            
        return nn.Dense(self.output_size)(x)