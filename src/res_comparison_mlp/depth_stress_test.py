# Путь: src/res_comparison_mlp/depth_stress_test.py

import os
import json
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from src.datasets import get_mnist_loaders, get_fashion_mnist_loaders
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
    one_hot_y = jax.nn.one_hot(batch_y, 10)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_y).mean()
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == batch_y)
    return loss, acc

def evaluate(state, test_loader):
    losses = []
    accs = []
    for data, target in test_loader:
        x_jax = jnp.array(data.numpy())
        y_jax = jnp.array(target.numpy())
        loss, acc = eval_step(state, x_jax, y_jax)
        losses.append(float(loss))
        accs.append(float(acc))
    return float(np.mean(losses)), float(np.mean(accs)) * 100

def run_experiment_for_dataset(dataset_name, train_loader, test_loader, depths, args):
    print("\n" + "="*60)
    print(f"НАЧАЛО ЭКСПЕРИМЕНТА ДЛЯ ДАТАСЕТА: {dataset_name.upper()}")
    print("="*60)
    
    dataset_results = {}
    block_types = ["random_skip", "standard", "no_skip"]
    
    for b_type in block_types:
        dataset_results[b_type] = {}
        for depth in depths:
            print(f"\n>>> Архитектура: {b_type.upper()} | Глубина (L): {depth} <<<")
            
            model = ComparativeResNetMLP(
                hidden_size=args.hidden_size,
                num_blocks=depth,
                output_size=10,
                block_type=b_type,
                seed=args.seed
            )
            
            rng = jax.random.PRNGKey(args.seed)
            dummy_shape = (args.batch_size, 28, 28, 1)
            state = create_train_state(model, rng, args.lr, dummy_shape)
            
            history = {
                "train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []
            }
            
            start_time = time.time()
            for epoch in range(args.epochs):
                epoch_loss = []
                epoch_acc = []
                for batch_idx, (data, target) in enumerate(train_loader):
                    x_jax = jnp.array(data.numpy())
                    y_jax = jnp.array(target.numpy())
                    
                    state, loss, acc = train_step(state, x_jax, y_jax)
                    epoch_loss.append(float(loss))
                    epoch_acc.append(float(acc))
                    
                train_loss_avg = float(np.mean(epoch_loss))
                train_acc_avg = float(np.mean(epoch_acc)) * 100
                test_loss_avg, test_acc_avg = evaluate(state, test_loader)
                
                history["train_loss"].append(train_loss_avg)
                history["train_acc"].append(train_acc_avg)
                history["test_loss"].append(test_loss_avg)
                history["test_acc"].append(test_acc_avg)
                
                if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
                    print(f"  Эпоха {epoch+1:02d}/{args.epochs:02d} | Loss: {train_loss_avg:.4f} | Train Acc: {train_acc_avg:.2f}% | Test Acc: {test_acc_avg:.2f}%")
            
            elapsed = time.time() - start_time
            print(f"Завершено за: {elapsed:.1f} сек. Итоговая точность: {history['test_acc'][-1]:.2f}%")
            
            dataset_results[b_type][str(depth)] = {
                "min_train_loss": float(np.min(history["train_loss"])),
                "final_train_loss": history["train_loss"][-1],
                "final_test_acc": history["test_acc"][-1],
                "best_test_acc": float(np.max(history["test_acc"])),
                "elapsed_time": elapsed,
                "history": history
            }
            
    return dataset_results

def main():
    parser = argparse.ArgumentParser(description="Запуск Эксперимента B: Предельная глубина обучаемости")
    parser.add_argument("--epochs", type=int, default=15, help="Количество эпох на конфигурацию")
    parser.add_argument("--hidden_size", type=int, default=128, help="Ширина скрытых слоев")
    parser.add_argument("--batch_size", type=int, default=128, help="Размер батча")
    parser.add_argument("--lr", type=float, default=0.001, help="Скорость обучения")
    parser.add_argument("--seed", type=int, default=42, help="Случайный сид")
    parser.add_argument("--depths", type=str, default="5,10,20,50,100", help="Глубины через запятую")
    parser.add_argument("--datasets", type=str, default="mnist,fashion_mnist", help="Датасеты для теста (через запятую)")
    parser.add_argument("--save_dir", type=str, default="results", help="Папка для сохранения результатов")
    args = parser.parse_args()

    depth_list = [int(x.strip()) for x in args.depths.split(",")]
    dataset_list = [x.strip().lower() for x in args.datasets.split(",")]
    
    payload = {
        "metadata": {
            "depths": depth_list,
            "hidden_size": args.hidden_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "batch_size": args.batch_size
        },
        "datasets": {}
    }
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    for dataset_name in dataset_list:
        if dataset_name == "mnist":
            train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
        elif dataset_name == "fashion_mnist":
            train_loader, test_loader = get_fashion_mnist_loaders(batch_size=args.batch_size)
        else:
            print(f"Неизвестный датасет: {dataset_name}. Пропуск.")
            continue
            
        dataset_results = run_experiment_for_dataset(
            dataset_name=dataset_name,
            train_loader=train_loader,
            test_loader=test_loader,
            depths=depth_list,
            args=args
        )
        payload["datasets"][dataset_name] = dataset_results

    # Сохранение собранных данных
    save_path = os.path.join(args.save_dir, "depth_stress_test.json")
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=4)
        
    print("\n" + "="*60)
    print(f"Эксперимент завершен. Данные сохранены по пути: {save_path}")
    print("="*60)

if __name__ == "__main__":
    main()