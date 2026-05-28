# Путь: src/res_comparison_mlp/train_comparison.py

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from src.models.residual_comparison import ComparativeResNetMLP

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
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == batch_y)
    return acc

def evaluate(state, test_loader):
    accs = []
    for data, target in test_loader:
        x_jax = jnp.array(data.numpy())
        y_jax = jnp.array(target.numpy())
        acc = eval_step(state, x_jax, y_jax)
        accs.append(acc)
    return float(np.mean(accs)) * 100

def run_single_train_experiment(block_type, train_loader, test_loader, args):
    print(f"\n--- Запуск обучения: {block_type.upper()} ---")
    
    model = ComparativeResNetMLP(
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
        output_size=10,
        block_type=block_type,
        seed=args.seed
    )
    
    rng = jax.random.PRNGKey(args.seed)
    dummy_shape = (128, 28, 28, 1)
    state = create_train_state(model, rng, args.lr, dummy_shape)
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            x_jax = jnp.array(data.numpy())
            y_jax = jnp.array(target.numpy())
            state, loss, acc = train_step(state, x_jax, y_jax)
            correct += int(float(acc) * len(target))
            total += len(target)
            
        train_acc = 100. * correct / total
        test_acc = evaluate(state, test_loader)
        best_acc = max(best_acc, test_acc)
        print(f"Epoch {epoch+1} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
        
    return best_acc

def run_training_comparison(train_loader, test_loader, args):
    results = {}
    for b_type in ["no_skip", "standard", "random_skip"]:
        results[b_type] = run_single_train_experiment(b_type, train_loader, test_loader, args)
        
    print("\n" + "="*50)
    print(f"РЕЗУЛЬТАТЫ СРАВНЕНИЯ ОБУЧЕНИЯ (Глубина: {args.num_blocks} блоков, Ширина: {args.hidden_size})")
    print("="*50)
    print(f"No Skip (Plain MLP)  : {results['no_skip']:.2f}%")
    print(f"Standard Skip (Ix)   : {results['standard']:.2f}%")
    print(f"Random Skip (Px)     : {results['random_skip']:.2f}%")
    print("="*50)
    return results