import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from src.datasets import get_adding_data
# Добавлен импорт новой модели SparseRandomTimeResModel
from src.models import StandardStackedRNN, Diagonal3TimeResModel, SparseRandomTimeResModel

def count_parameters(state):
    return sum(x.size for x in jax.tree_util.tree_leaves(state.params))

def create_train_state(model, rng, lr, dummy_shape):
    params = model.init(rng, jnp.ones(dummy_shape))['params']
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr)
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_x, batch_y):
    def loss_fn(params):
        preds = state.apply_fn({'params': params}, batch_x)
        loss = jnp.mean((preds.squeeze(-1) - batch_y.squeeze(-1)) ** 2)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train_adding(model, n_iters=2000, batch_size=64, seq_len=50, lr=0.001):
    rng = jax.random.PRNGKey(42)
    dummy_shape = (batch_size, seq_len, 2)
    state = create_train_state(model, rng, lr, dummy_shape)
    
    print(f"Parameters: {count_parameters(state)}")

    for i in range(n_iters):
        # Загрузка данных (возвращает PyTorch Tensors)
        inputs, targets = get_adding_data(batch_size, seq_len)
        
        # Конвертация в JAX
        x_jax = jnp.array(inputs.numpy())
        y_jax = jnp.array(targets.numpy())

        state, loss = train_step(state, x_jax, y_jax)

        if i % 100 == 0:
            print(f"Iter {i}, Loss: {float(loss):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNNs on the Adding Problem in JAX")
    # Добавлен вариант "sparse_random" в choices
    parser.add_argument("--model", type=str, default="diagonal", choices=["standard", "diagonal", "sparse_random"])
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--iters", type=int, default=2000)
    # Новые аргументы для настройки параметров разреженного случайного графа
    parser.add_argument("--target_diameter", type=float, default=3.0, help="Target diameter of the connection graph")
    parser.add_argument("--k", type=int, default=None, help="Directly set number of random connections per node (overrides target_diameter)")
    args = parser.parse_args()

    # Логика выбора и инициализации модели
    if args.model == "standard":
        model = StandardStackedRNN(input_size=2, hidden_size=60, num_layers=args.num_layers)
    elif args.model == "diagonal":
        model = Diagonal3TimeResModel(input_size=2, hidden_size=args.hidden_size, num_layers=args.num_layers)
    elif args.model == "sparse_random":
        model = SparseRandomTimeResModel(
            input_size=2,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            target_diameter=args.target_diameter,
            k=args.k
        )

    print(f"Model: {args.model.upper()}")
    print(f"Sequence Length: {args.seq_len}")
    train_adding(model, n_iters=args.iters, seq_len=args.seq_len)