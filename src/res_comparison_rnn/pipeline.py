# Путь: src/res_comparison_rnn/pipeline.py

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
from src.datasets import get_adding_data
from src.models import ComparativeStackedRNN

def set_global_seeds(seed: int):
    """Фиксирует случайность во всех используемых библиотеках."""
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

# --- JIT-компиляция шагов для классификации (MNIST / Fashion MNIST) ---
@jax.jit
def train_step_class(state, batch_x, batch_y):
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
def eval_step_class(state, batch_x, batch_y):
    logits = state.apply_fn({'params': state.params}, batch_x)
    one_hot_y = jax.nn.one_hot(batch_y, 10)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_y).mean()
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == batch_y)
    return loss, acc

# --- JIT-компиляция шагов для регрессии (Adding Problem) ---
@jax.jit
def train_step_reg(state, batch_x, batch_y):
    def loss_fn(params):
        preds = state.apply_fn({'params': params}, batch_x)
        loss = jnp.mean((preds.squeeze(-1) - batch_y.squeeze(-1)) ** 2)
        return loss, preds

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, preds), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step_reg(state, batch_x, batch_y):
    preds = state.apply_fn({'params': state.params}, batch_x)
    loss = jnp.mean((preds.squeeze(-1) - batch_y.squeeze(-1)) ** 2)
    return loss

# --- Расчет и анализ градиентного потока по слоям ---
def compute_rnn_gradients(state, batch_x, batch_y, is_regression):
    if is_regression:
        def loss_fn(params):
            preds = state.apply_fn({'params': params}, batch_x)
            return jnp.mean((preds.squeeze(-1) - batch_y.squeeze(-1)) ** 2)
    else:
        def loss_fn(params):
            logits = state.apply_fn({'params': params}, batch_x)
            one_hot_y = jax.nn.one_hot(batch_y, 10)
            return optax.softmax_cross_entropy(logits=logits, labels=one_hot_y).mean()

    grad_fn = jax.value_and_grad(loss_fn)
    _, grads = grad_fn(state.params)
    return grads

def analyze_gradient_flow_rnn(state, grads):
    """
    Рассчитывает относительные нормы градиентов по весам слоев.
    """
    params = state.params
    layer_stats = []
    
    layer_keys = [k for k in params.keys() if k.startswith("layers_")]
    layer_keys = sorted(layer_keys, key=lambda k: int(k.split("_")[1]))
    
    for key in layer_keys:
        layer_idx = int(key.split("_")[1])
        layer_params = params[key]
        layer_grads = grads[key]
        
        stats = {"layer_idx": layer_idx}
        for sub in ['w_up', 'w_rec']:
            if sub in layer_params:
                w = layer_params[sub]['kernel']
                dw = layer_grads[sub]['kernel']
                
                w_norm = float(jnp.linalg.norm(w))
                dw_norm = float(jnp.linalg.norm(dw))
                rel_grad = dw_norm / w_norm if w_norm > 0 else 0.0
                
                stats[f"{sub}_weight_norm"] = w_norm
                stats[f"{sub}_grad_norm"] = dw_norm
                stats[f"{sub}_relative_grad"] = rel_grad
                
        layer_stats.append(stats)
    return layer_stats

def run_rnn_pipeline(
    model_type: str,
    depth: int,
    hidden_size: int,
    dataset_name: str,
    train_loader,
    test_loader,
    epochs: int,
    lr: float,
    batch_size: int,
    seq_len: int,
    seed: int,
    save_dir: str
):
    # Фиксируем воспроизводимость для текущей конфигурации
    set_global_seeds(seed)
    is_regression = (dataset_name == "adding")
    
    print(f"\n>>> Запуск RNN: {model_type.upper()} | L={depth} | W={hidden_size} | {dataset_name.upper()} (Seed: {seed}) <<<")
    
    input_size = 2 if is_regression else 1
    output_size = 1 if is_regression else 10
    
    model = ComparativeStackedRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=depth,
        seq_len=seq_len,
        output_size=output_size,
        model_type=model_type,
        seed=seed
    )
    
    rng = jax.random.PRNGKey(seed)
    dummy_shape = (batch_size, seq_len, input_size)
    state = create_train_state(model, rng, lr, dummy_shape)
    
    # 1. Сбор опорных данных для вычисления градиентов
    if is_regression:
        ref_x, ref_y = get_adding_data(batch_size=batch_size, seq_len=seq_len)
        ref_x_jax = jnp.array(ref_x.numpy())
        ref_y_jax = jnp.array(ref_y.numpy())
        
        # Фиксированный набор тестов для Adding Problem
        set_global_seeds(seed + 1000)
        val_inputs, val_targets = get_adding_data(batch_size=2000, seq_len=seq_len)
        test_x_jax = jnp.array(val_inputs.numpy())
        test_y_jax = jnp.array(val_targets.numpy())
        set_global_seeds(seed)  # Возврат к основному сиду
    else:
        ref_batch_x, ref_batch_y = next(iter(train_loader))
        ref_batch_x = ref_batch_x.view(-1, 784, 1)
        ref_x_jax = jnp.array(ref_batch_x.numpy())
        ref_y_jax = jnp.array(ref_batch_y.numpy())
        
    # Снимаем градиентный поток на инициализации
    init_grads = compute_rnn_gradients(state, ref_x_jax, ref_y_jax, is_regression)
    gradient_flow_init = analyze_gradient_flow_rnn(state, init_grads)
    
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
        
        if is_regression:
            # Одна "эпоха" Adding Problem = 100 генерируемых итераций
            for step in range(100):
                inputs, targets = get_adding_data(batch_size=batch_size, seq_len=seq_len)
                x_jax = jnp.array(inputs.numpy())
                y_jax = jnp.array(targets.numpy())
                state, loss = train_step_reg(state, x_jax, y_jax)
                epoch_loss.append(float(loss))
                
            train_loss_avg = float(np.mean(epoch_loss))
            test_loss_avg = float(eval_step_reg(state, test_x_jax, test_y_jax))
            
            history["train_loss"].append(train_loss_avg)
            history["train_acc"].append(0.0)
            history["test_loss"].append(test_loss_avg)
            history["test_acc"].append(0.0)
            
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"  [Эпоха {epoch+1:02d}/{epochs:02d}] Train MSE: {train_loss_avg:.5f} | Test MSE: {test_loss_avg:.5f}")
        else:
            # Классификация (MNIST / Fashion MNIST)
            for data, target in train_loader:
                data = data.view(-1, 784, 1)
                x_jax = jnp.array(data.numpy())
                y_jax = jnp.array(target.numpy())
                state, loss, acc = train_step_class(state, x_jax, y_jax)
                epoch_loss.append(float(loss))
                epoch_acc.append(float(acc))
                
            train_loss_avg = float(np.mean(epoch_loss))
            train_acc_avg = float(np.mean(epoch_acc)) * 100
            
            test_losses = []
            test_accs = []
            for data, target in test_loader:
                data = data.view(-1, 784, 1)
                x_jax = jnp.array(data.numpy())
                y_jax = jnp.array(target.numpy())
                t_loss, t_acc = eval_step_class(state, x_jax, y_jax)
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
    metric_str = f"MSE: {history['test_loss'][-1]:.5f}" if is_regression else f"Test Acc: {history['test_acc'][-1]:.2f}%"
    print(f"  Обучение завершено за {elapsed_time:.1f} сек. Итог: {metric_str}")
    
    # Снимаем градиентный поток в конце обучения
    final_grads = compute_rnn_gradients(state, ref_x_jax, ref_y_jax, is_regression)
    gradient_flow_final = analyze_gradient_flow_rnn(state, final_grads)
    
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
            "seq_len": seq_len,
            "seed": seed,
            "elapsed_time": elapsed_time,
            "is_regression": is_regression
        },
        "history": history,
        "gradient_flow_init": gradient_flow_init,
        "gradient_flow_final": gradient_flow_final,
        "params": cpu_params
    }
    
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{dataset_name}_{model_type}_L{depth}_W{hidden_size}_seq{seq_len}_seed{seed}.pkl"
    save_path = os.path.join(save_dir, filename)
    
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
        
    print(f"  Результаты сохранены в: {save_path}")
    return payload