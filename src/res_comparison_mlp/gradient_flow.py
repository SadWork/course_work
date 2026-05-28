# Путь: src/res_comparison_mlp/gradient_flow.py

import os
import json
import argparse
import jax
import jax.numpy as jnp
import optax

from src.datasets import get_mnist_loaders
from src.models.residual_comparison import ComparativeResNetMLP

def run_gradient_flow_experiment(train_loader, args):
    """
    Выполняет Эксперимент А: анализ градиентного затухания при инициализации.
    Замеряет относительные нормы градиентов по весам для каждого из N блоков
    для трех конфигураций: 'no_skip', 'standard', 'random_skip'.
    """
    print("\n" + "="*60)
    print("ЗАПУСК ЭКСПЕРИМЕНТА А: ГРАДИЕНТНЫЙ ПОТОК ПРИ ИНИЦИАЛИЗАЦИИ")
    print("="*60)
    
    results = {}
    block_types = ["no_skip", "standard", "random_skip"]
    
    # Извлечение одного батча для оценки
    batch_x, batch_y = next(iter(train_loader))
    x_jax = jnp.array(batch_x.numpy())
    y_jax = jnp.array(batch_y.numpy())
    
    for b_type in block_types:
        print(f"Расчет градиентов для архитектуры: {b_type.upper()}")
        
        # Инициализация модели со случайными весами
        model = ComparativeResNetMLP(
            hidden_size=args.hidden_size,
            num_blocks=args.num_blocks,
            output_size=10,
            block_type=b_type,
            seed=args.seed
        )
        
        rng = jax.random.PRNGKey(args.seed)
        dummy_shape = (x_jax.shape[0], 28, 28, 1)
        variables = model.init(rng, jnp.ones(dummy_shape))
        params = variables['params']
        
        # Функция потерь на одном батче
        @jax.jit
        def loss_fn(p):
            logits = model.apply({'params': p}, x_jax)
            one_hot_y = jax.nn.one_hot(y_jax, 10)
            return optax.softmax_cross_entropy(logits=logits, labels=one_hot_y).mean()
            
        # Нахождение градиентов по всем параметрам
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        
        layer_stats = []
        for i in range(args.num_blocks):
            block_name = f"ComparativeResBlock_{i}"
            if block_name in params:
                block_params = params[block_name]
                block_grads = grads[block_name]
                
                # Поиск ключа полносвязного слоя внутри блока
                dense_key = None
                for k in block_params.keys():
                    if k.lower().startswith('dense'):
                        dense_key = k
                        break
                        
                if dense_key is not None:
                    w = block_params[dense_key]['kernel']
                    dw = block_grads[dense_key]['kernel']
                    
                    # Вычисление нормы Фробениуса
                    w_norm = float(jnp.linalg.norm(w))
                    dw_norm = float(jnp.linalg.norm(dw))
                    
                    # Относительная норма градиента: G_l = ||dw||_2 / ||w||_2
                    rel_grad = dw_norm / w_norm if w_norm > 0 else 0.0
                    
                    layer_stats.append({
                        "layer_idx": i,
                        "weight_norm": w_norm,
                        "grad_norm": dw_norm,
                        "relative_grad": rel_grad
                    })
                    
        results[b_type] = layer_stats
        print(f"Успешно обработано блоков: {len(layer_stats)}")
        if layer_stats:
            print(f" -> Первый блок ({layer_stats[0]['layer_idx']}) rel_grad: {layer_stats[0]['relative_grad']:.2e}")
            print(f" -> Последний блок ({layer_stats[-1]['layer_idx']}) rel_grad: {layer_stats[-1]['relative_grad']:.2e}")

    # Сборка финального JSON-файла результатов
    payload = {
        "metadata": {
            "num_blocks": args.num_blocks,
            "hidden_size": args.hidden_size,
            "seed": args.seed,
            "batch_size": x_jax.shape[0]
        },
        "results": results
    }
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "gradient_flow_init.json")
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=4)
        
    print("\n" + "="*60)
    print(f"Эксперимент завершен. Метрики сохранены по пути: {save_path}")
    print("="*60)
    return payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск Эксперимента А: Градиентный поток при инициализации")
    parser.add_argument("--num_blocks", type=int, default=20, help="Глубина сети (число блоков)")
    parser.add_argument("--hidden_size", type=int, default=128, help="Ширина скрытых слоев")
    parser.add_argument("--batch_size", type=int, default=128, help="Размер батча для прохода")
    parser.add_argument("--seed", type=int, default=42, help="Случайный сид")
    parser.add_argument("--save_dir", type=str, default="results", help="Директория сохранения результатов")
    args = parser.parse_args()

    # Загрузка MNIST
    train_loader, _ = get_mnist_loaders(batch_size=args.batch_size)

    # Запуск эксперимента напрямую
    run_gradient_flow_experiment(train_loader, args)