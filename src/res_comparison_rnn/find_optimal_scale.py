# Путь: src/res_comparison_rnn/find_optimal_scale.py

import argparse
import itertools
import numpy as np
import jax
import jax.numpy as jnp
import optax
from src.datasets import get_adding_data
from src.models import ComparativeStackedRNN

def parse_list_arg(arg_str, target_type):
    return [target_type(x.strip()) for x in arg_str.split(",") if x.strip()]

def compute_gradients_for_scale(scale_val, model_type, depth, hidden_size, seq_len, x_jax, y_jax):
    """
    Вычисляет средний профиль градиентов по скрытым состояниям.
    """
    model = ComparativeStackedRNN(
        input_size=x_jax.shape[-1],
        hidden_size=hidden_size,
        num_layers=depth,
        seq_len=seq_len,
        output_size=1,
        model_type=model_type,
        custom_scale=scale_val,
        seed=42
    )
    
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, x_jax)['params']
    
    num_layers = depth
    batch_size = x_jax.shape[0]
    init_perturbations = jnp.zeros((num_layers, seq_len, batch_size, hidden_size))

    def loss_fn(perturbations):
        preds = model.apply({'params': params}, x_jax, perturbations=perturbations)
        loss = jnp.mean((preds.squeeze(-1) - y_jax.squeeze(-1)) ** 2)
        return loss

    grad_fn = jax.grad(loss_fn)
    state_grads = grad_fn(init_perturbations)  # (L, T, B, H)
    
    state_grad_norms = jnp.linalg.norm(state_grads, axis=-1)  # (L, T, B)
    mean_state_grad_norms = jnp.mean(state_grad_norms, axis=-1)  # (L, T)
    
    return mean_state_grad_norms

def search_optimal_scale(model_type, depth, hidden_size, seq_len, x_jax, y_jax, args):
    """
    Двухфазный гибридный поиск:
    1. Равномерное сканирование (Grid Search) для локализации глобального минимума.
    2. Локальный бинарный поиск в окрестности наилучшей точки.
    """
    low = args.scale_low
    high = args.scale_high
    grid_steps = args.grid_steps
    
    print(f"Поиск для L={depth} | W={hidden_size} | T={seq_len}...")
    
    # ================= ФАЗА 1: РАВНОМЕРНОЕ СКАНИРОВАНИЕ =================
    print(f"  [Фаза 1] Равномерное сканирование ({grid_steps} точек в диапазоне [{low}, {high}])...")
    grid_scales = np.linspace(low, high, grid_steps)
    grid_results = []
    
    for k, scale in enumerate(grid_scales):
        try:
            mean_grads = compute_gradients_for_scale(scale, model_type, depth, hidden_size, seq_len, x_jax, y_jax)
            
            log_ratios = []
            for l in range(depth):
                g_start = mean_grads[l, 0]
                g_end = mean_grads[l, -1]
                log_ratio = jnp.log(g_start + 1e-35) - jnp.log(g_end + 1e-35)
                log_ratios.append(log_ratio)
                
            avg_log_ratio = float(np.mean(log_ratios))
            
            if np.isnan(avg_log_ratio) or np.isinf(avg_log_ratio):
                metric = float('inf')
            else:
                metric = abs(avg_log_ratio)
                
            grid_results.append((scale, avg_log_ratio, metric))
            print(f"    Точка {k+1:02d}/{grid_steps:02d} | scale={scale:.5f} | log_ratio={avg_log_ratio:.4f} | ratio={np.exp(avg_log_ratio):.2e}")
            
        except Exception as e:
            grid_results.append((scale, float('inf'), float('inf')))
            print(f"    Точка {k+1:02d}/{grid_steps:02d} | scale={scale:.5f} | Ошибка вычислений / NaN")
            
    # Выбираем индекс точки с минимальным отклонением логарифма от нуля
    best_idx = np.argmin([r[2] for r in grid_results])
    best_grid_scale, best_grid_log_ratio, _ = grid_results[best_idx]
    
    print(f"  [Итог Фазы 1] Наилучшая начальная точка: scale={best_grid_scale:.5f} (log_ratio={best_grid_log_ratio:.4f})")
    
    # Определяем суженный диапазон для Фазы 2
    local_low = grid_scales[max(0, best_idx - 1)]
    local_high = grid_scales[min(grid_steps - 1, best_idx + 1)]
    
    # Корректировка границ, если лучшая точка на краях сетки
    if best_idx == 0:
        local_low = grid_scales[0]
        local_high = grid_scales[1]
    elif best_idx == grid_steps - 1:
        local_low = grid_scales[-2]
        local_high = grid_scales[-1]
        
    # ================= ФАЗА 2: ЛОКАЛЬНЫЙ БИНАРНЫЙ ПОИСК =================
    print(f"  [Фаза 2] Локальный бинарный поиск в интервале [{local_low:.5f}, {local_high:.5f}]...")
    
    best_scale = best_grid_scale
    best_ratio_diff = abs(best_grid_log_ratio)
    best_final_ratio = np.exp(best_grid_log_ratio)
    
    low_b = local_low
    high_b = local_high
    
    for i in range(args.max_iters):
        mid = (low_b + high_b) / 2.0
        
        try:
            mean_grads = compute_gradients_for_scale(mid, model_type, depth, hidden_size, seq_len, x_jax, y_jax)
            
            log_ratios = []
            for l in range(depth):
                g_start = mean_grads[l, 0]
                g_end = mean_grads[l, -1]
                log_ratio = jnp.log(g_start + 1e-35) - jnp.log(g_end + 1e-35)
                log_ratios.append(log_ratio)
                
            avg_log_ratio = float(np.mean(log_ratios))
            avg_ratio = np.exp(avg_log_ratio)
            
            print(f"    Итерация {i+1:02d} | scale={mid:.5f} | log_ratio={avg_log_ratio:.4f} | ratio={avg_ratio:.2e}")
            
            if np.isnan(avg_log_ratio) or np.isinf(avg_log_ratio):
                high_b = mid
                continue
                
            if abs(avg_log_ratio) < best_ratio_diff:
                best_ratio_diff = abs(avg_log_ratio)
                best_scale = mid
                best_final_ratio = avg_ratio
                
            if avg_log_ratio < 0.0:
                # Сигнал затухает -> увеличиваем диапазон весов
                low_b = mid
            else:
                # Сигнал взрывается -> уменьшаем диапазон весов
                high_b = mid
                
            if abs(avg_log_ratio) < args.tol:
                print(f"    Сходимость достигнута по критерию tol={args.tol}")
                break
                
        except Exception as e:
            high_b = mid
            continue
            
    return best_scale, best_final_ratio

def main():
    parser = argparse.ArgumentParser(description="Гибридный поиск оптимального scale_val для RNN")
    parser.add_argument("--models", type=str, default="random_balanced", 
                        help="Типы моделей через запятую (random_balanced, random_uniform)")
    parser.add_argument("--depths", type=str, default="2,4", help="Список глубин через запятую")
    parser.add_argument("--hidden_sizes", type=str, default="16,64", help="Список скрытых слоев через запятую")
    parser.add_argument("--seq_lens", type=str, default="100,500,1000", help="Список длин серий")
    parser.add_argument("--scale_low", type=float, default=0.01, help="Нижняя граница поиска")
    parser.add_argument("--scale_high", type=float, default=5.0, help="Верхняя граница поиска")
    parser.add_argument("--grid_steps", type=int, default=10, help="Количество равномерных шагов на Фазе 1")
    parser.add_argument("--max_iters", type=int, default=15, help="Количество шагов бинарного поиска на Фазе 2")
    parser.add_argument("--tol", type=float, default=0.01, help="Допустимое отклонение log_ratio от 0")
    parser.add_argument("--batch_size", type=int, default=128, help="Размер батча")
    args = parser.parse_args()

    model_list = parse_list_arg(args.models, str)
    depth_list = parse_list_arg(args.depths, int)
    hidden_size_list = parse_list_arg(args.hidden_sizes, int)
    seq_len_list = parse_list_arg(args.seq_lens, int)

    grid = list(itertools.product(model_list, depth_list, hidden_size_list, seq_len_list))
    results = []
    
    print("=" * 80)
    print(f"СТАРТ СЕТКИ ГИБРИДНЫХ ЭКСПЕРИМЕНТОВ (Coarse-to-Fine)")
    print(f"Всего конфигураций: {len(grid)}")
    print("=" * 80)

    for model_type, depth, hidden_size, seq_len in grid:
        print(f"\n[Конфигурация] {model_type.upper()} | L={depth} | W={hidden_size} | T={seq_len}")
        
        inputs, targets = get_adding_data(batch_size=args.batch_size, seq_len=seq_len)
        x_jax = jnp.array(inputs.numpy())
        y_jax = jnp.array(targets.numpy())
        
        opt_scale, opt_ratio = search_optimal_scale(
            model_type=model_type,
            depth=depth,
            hidden_size=hidden_size,
            seq_len=seq_len,
            x_jax=x_jax,
            y_jax=y_jax,
            args=args
        )
        
        results.append({
            "model": model_type,
            "L": depth,
            "W": hidden_size,
            "T": seq_len,
            "opt_scale": opt_scale,
            "ratio": opt_ratio
        })
        
        print(f"-> Оптимальный scale_val: {opt_scale:.5f} (Итоговое отношение: {opt_ratio:.4e})")

    # Вывод результатов
    print("\n" + "="*90)
    print("ИТОГОВАЯ ТАБЛИЦА ЗАВИСИМОСТИ SCALE_VAL ДЛЯ ОТЧЕТА (COARSE-TO-FINE)")
    print("="*90)
    print(f"| {'Архитектура':<15} | {'Глубина L':<9} | {'Ширина W':<8} | {'Длина T':<7} | {'Оптим. scale_val':<18} | {'Итог. Ratio':<12} |")
    print(f"|{'-'*17}|{'-'*11}|{'-'*10}|{'-'*9}|{'-'*20}|{'-'*14}|")
    for r in results:
        print(f"| {r['model']:<15} | {r['L']:<9} | {r['W']:<8} | {r['T']:<7} | {r['opt_scale']:<18.5f} | {r['ratio']:<12.2e} |")
    print("="*90 + "\n")

if __name__ == "__main__":
    main()