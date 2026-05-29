# Путь: src/res_comparison_mlp/pipeline.py

import os
import random
import pickle
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax.training import train_state
from src.models.residual_comparison import ComparativeResNetMLP

def set_global_seeds(seed: int):
    """
    Устанавливает глобальные сиды для всех библиотек, участвующих в процессе,
    обеспечивая детерминизм порядка батчей в DataLoader и инициализации весов.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def compute_block_gradients(state, batch_x, batch_y):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch_x)
        one_hot_y = jax.nn.one_hot(batch_y, 10)
        return optax.softmax_cross_entropy(logits=logits, labels=one_hot_y).mean()

    grad_fn = jax.value_and_grad(loss_fn)
    _, grads = grad_fn(state.params)
    return grads

def analyze_gradient_flow(state, grads):
    params = state.params
    layer_stats = []
    
    block_keys = [k for k in params.keys() if k.startswith("ComparativeResBlock_")]
    block_keys = sorted(block_keys, key=lambda k: int(k.split("_")[1]))
    
    for key in block_keys:
        block_idx = int(key.split("_")[1])
        block_params = params[key]
        block_grads = grads[key]
        
        dense_key = None
        for k in block_params.keys():
            if k.lower().startswith('dense'):
                dense_key = k
                break
                
        if dense_key is not None:
            w = block_params[dense_key]['kernel']
            dw = block_grads[dense_key]['kernel']
            
            w_norm = float(jnp.linalg.norm(w))
            dw_norm = float(jnp.linalg.norm(dw))
            rel_grad = dw_norm / w_norm if w_norm > 0 else 0.0
            
            layer_stats.append({
                "block_idx": block_idx,
                "weight_norm": w_norm,
                "grad_norm": dw_norm,
                "relative_grad": rel_grad
            })
            
    return layer_stats

def run_training_pipeline(
    model_type: str,
    depth: int,
    hidden_size: int,
    dataset_name: str,
    train_loader,
    test_loader,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    save_dir: str
):
    """
    Запускает детерминированный и воспроизводимый цикл обучения.
    """
    # 1. Сброс глобальных генераторов перед стартом каждого эксперимента
    set_global_seeds(seed)
    
    print(f"Активные устройства JAX для данного запуска: {jax.devices()}")
    print(f"\n>>> Запуск: {model_type.upper()} | L={depth} | W={hidden_size} | {dataset_name.upper()} (Seed: {seed}) <<<")
    
    model = ComparativeResNetMLP(
        hidden_size=hidden_size,
        num_blocks=depth,
        output_size=10,
        block_type=model_type,
        seed=seed
    )
    
    # Инициализация JAX-состояния с воспроизводимым ключом
    rng = jax.random.PRNGKey(seed)
    dummy_shape = (batch_size, 28, 28, 1)
    state = create_train_state(model, rng, lr, dummy_shape)
    
    # Извлекаем опорный батч для анализа градиентного потока
    ref_batch_x, ref_batch_y = next(iter(train_loader))
    ref_x_jax = jnp.array(ref_batch_x.numpy())
    ref_y_jax = jnp.array(ref_batch_y.numpy())
    
    # Анализ градиентного потока на инициализации
    init_grads = compute_block_gradients(state, ref_x_jax, ref_y_jax)
    gradient_flow_init = analyze_gradient_flow(state, init_grads)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = []
        epoch_acc = []
        
        for data, target in train_loader:
            x_jax = jnp.array(data.numpy())
            y_jax = jnp.array(target.numpy())
            state, loss, acc = train_step(state, x_jax, y_jax)
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))
            
        train_loss_avg = float(np.mean(epoch_loss))
        train_acc_avg = float(np.mean(epoch_acc)) * 100
        
        test_losses = []
        test_accs = []
        for data, target in test_loader:
            x_jax = jnp.array(data.numpy())
            y_jax = jnp.array(target.numpy())
            t_loss, t_acc = eval_step(state, x_jax, y_jax)
            test_losses.append(float(t_loss))
            test_accs.append(float(t_acc))
            
        test_loss_avg = float(np.mean(test_losses))
        test_acc_avg = float(np.mean(test_accs)) * 100
        
        history["train_loss"].append(train_loss_avg)
        history["train_acc"].append(train_acc_avg)
        history["test_loss"].append(test_loss_avg)
        history["test_acc"].append(test_acc_avg)
        
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"  [Эпоха {epoch+1:02d}/{epochs:02d}] Loss: {train_loss_avg:.4f} | Train Acc: {train_acc_avg:.2f}% | Test Acc: {test_acc_avg:.2f}%")
            
    elapsed_time = time.time() - start_time
    print(f"  Обучение завершено за {elapsed_time:.1f} сек. Итоговая точность: {history['test_acc'][-1]:.2f}%")
    
    # Анализ градиентного потока на уже обученной сети
    final_grads = compute_block_gradients(state, ref_x_jax, ref_y_jax)
    gradient_flow_final = analyze_gradient_flow(state, final_grads)
    
    cpu_params = jax.device_get(state.params)
    
    payload = {
        "metadata": {
            "model_type": model_type,
            "depth": depth,
            "hidden_size": hidden_size,
            "dataset_name": dataset_name,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "seed": seed,
            "elapsed_time": elapsed_time
        },
        "history": history,
        "gradient_flow_init": gradient_flow_init,
        "gradient_flow_final": gradient_flow_final,
        "params": cpu_params
    }
    
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{dataset_name}_{model_type}_L{depth}_W{hidden_size}_lr{lr}_epochs{epochs}_seed{seed}.pkl"
    save_path = os.path.join(save_dir, filename)
    
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
        
    print(f"  Результаты сохранены в: {save_path}")
    return payload