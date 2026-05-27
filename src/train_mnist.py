import argparse
import numpy as np
import torch
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from src.datasets import get_mnist_loaders
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
        logits = state.apply_fn({'params': params}, batch_x)
        one_hot_y = jax.nn.one_hot(batch_y, 10)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_y).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == batch_y)
    return state, loss, acc

@jax.jit
def eval_step(state, batch_x, batch_y):
    logits = state.apply_fn({'params': state.params}, batch_x)
    one_hot_y = jax.nn.one_hot(batch_y, 10)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_y).mean()
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == batch_y)
    return loss, acc

def train_classification(model, train_loader, test_loader, epochs=3, lr=0.001, perm_idx=None, seq_len=784):
    rng = jax.random.PRNGKey(42)
    dummy_shape = (128, seq_len, 1)
    state = create_train_state(model, rng, lr, dummy_shape)
    
    print(f"Parameters: {count_parameters(state)}")

    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, seq_len, 1)
            if perm_idx is not None:
                data = data[:, perm_idx, :]

            # Преобразование PyTorch Tensors -> JAX Arrays
            x_jax = jnp.array(data.numpy())
            y_jax = jnp.array(target.numpy())

            state, loss, acc = train_step(state, x_jax, y_jax)
            
            running_loss += float(loss)
            correct += int(float(acc) * len(target))
            total += len(target)

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {float(loss):.4f} | Accuracy: {float(acc) * 100:.2f}%")

        test_acc = evaluate_classification(state, test_loader, perm_idx, seq_len)
        train_acc = 100. * correct / total
        print(f"=== Epoch {epoch+1} Complete. Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}% ===\n")

def evaluate_classification(state, test_loader, perm_idx=None, seq_len=784):
    accs = []
    for data, target in test_loader:
        data = data.view(-1, seq_len, 1)
        if perm_idx is not None:
            data = data[:, perm_idx, :]
        x_jax = jnp.array(data.numpy())
        y_jax = jnp.array(target.numpy())
        
        _, acc = eval_step(state, x_jax, y_jax)
        accs.append(acc)
    return float(np.mean(accs)) * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNNs on psMNIST in JAX")
    parser.add_argument("--model", type=str, default="diagonal", choices=["standard", "diagonal", "sparse_random"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=191)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    # Новые аргументы для настройки графа
    parser.add_argument("--target_diameter", type=float, default=3.0, help="Target diameter of the connection graph")
    parser.add_argument("--k", type=int, default=None, help="Directly set number of random connections per node (overrides target_diameter)")
    args = parser.parse_args()

    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)

    seq_len = 784
    torch.manual_seed(42)
    pixel_permutation = torch.randperm(seq_len)

    if args.model == "standard":
        model = StandardStackedRNN(input_size=1, hidden_size=128, num_layers=4, output_size=10)
    elif args.model == "diagonal":
        model = Diagonal3TimeResModel(input_size=1, hidden_size=args.hidden_size, num_layers=args.num_layers, output_size=10)
    elif args.model == "sparse_random":
        model = SparseRandomTimeResModel(
            input_size=1, 
            hidden_size=args.hidden_size, 
            num_layers=args.num_layers, 
            output_size=10,
            target_diameter=args.target_diameter,
            k=args.k
        )

    print(f"Model: {args.model.upper()}")
    train_classification(model, train_loader, test_loader, epochs=args.epochs, perm_idx=pixel_permutation, seq_len=seq_len)