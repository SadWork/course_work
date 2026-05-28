# Путь: src/res_comparison_rnn/runner.py

import argparse
import itertools
from src.datasets import get_mnist_loaders, get_fashion_mnist_loaders
from src.res_comparison_rnn.pipeline import run_rnn_pipeline, set_global_seeds

def parse_list_arg(arg_str, target_type):
    return [target_type(x.strip()) for x in arg_str.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser(description="Автоматический запуск сетки экспериментов RNN")
    parser.add_argument("--models", type=str, default="standard,spatial,temporal,alternating,random_balanced,random_uniform",
                        help="Архитектуры RNN через запятую (standard, spatial, temporal, alternating, random_balanced, random_uniform)")
    parser.add_argument("--depths", type=str, default="2,4,8",
                        help="Список глубин (числа слоев) через запятую")
    parser.add_argument("--hidden_sizes", type=str, default="64",
                        help="Список ширин скрытых слоев через запятую")
    parser.add_argument("--datasets", type=str, default="mnist,fashion_mnist,adding",
                        help="Список датасетов через запятую (mnist, fashion_mnist, adding)")
    parser.add_argument("--epochs", type=int, default=10, help="Количество эпох на конфигурацию")
    parser.add_argument("--lr", type=float, default=0.001, help="Скорость обучения (Adam)")
    parser.add_argument("--batch_size", type=int, default=128, help="Размер батча")
    parser.add_argument("--seq_len_adding", type=int, default=50, help="Длина последовательности для Adding Problem")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно для воспроизводимости")
    parser.add_argument("--save_dir", type=str, default="results_rnn_pkl", help="Директория сохранения результатов")
    args = parser.parse_args()

    # Фиксируем seed для воспроизводимой загрузки данных
    set_global_seeds(args.seed)

    model_list = parse_list_arg(args.models, str)
    depth_list = parse_list_arg(args.depths, int)
    hidden_size_list = parse_list_arg(args.hidden_sizes, int)
    dataset_list = parse_list_arg(args.datasets, str)

    experiments = list(itertools.product(dataset_list, model_list, depth_list, hidden_size_list))
    
    print("=" * 80)
    print(f"Запланировано RNN экспериментов: {len(experiments)}")
    print(f"Архитектуры:  {model_list}")
    print(f"Глубины (L):  {depth_list}")
    print(f"Ширины (W):   {hidden_size_list}")
    print(f"Датасеты:     {dataset_list}")
    print(f"Зерно (Seed): {args.seed}")
    print("=" * 80)

    loaders_cache = {}

    for idx, (dataset_name, model_type, depth, hidden_size) in enumerate(experiments):
        print(f"\n[RNN Эксперимент {idx + 1}/{len(experiments)}]")
        
        # Определяем физическую длину временного ряда
        seq_len = args.seq_len_adding if dataset_name == "adding" else 784
        
        # Загрузка классификационных сетов (Adding генерируется динамически)
        train_loader, test_loader = None, None
        if dataset_name != "adding":
            if dataset_name not in loaders_cache:
                print(f"Загрузка датасета: {dataset_name.upper()}...")
                set_global_seeds(args.seed)
                if dataset_name == "mnist":
                    loaders_cache[dataset_name] = get_mnist_loaders(batch_size=args.batch_size)
                elif dataset_name == "fashion_mnist":
                    loaders_cache[dataset_name] = get_fashion_mnist_loaders(batch_size=args.batch_size)
                else:
                    print(f"Пропуск неизвестного датасета '{dataset_name}'")
                    continue
            train_loader, test_loader = loaders_cache[dataset_name]

        # Запуск пайплайна обучения
        try:
            run_rnn_pipeline(
                model_type=model_type,
                depth=depth,
                hidden_size=hidden_size,
                dataset_name=dataset_name,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                seq_len=seq_len,
                seed=args.seed,
                save_dir=args.save_dir
            )
        except Exception as e:
            print(f"Ошибка при выполнении конфигурации rnn_{model_type} L{depth} W{hidden_size}: {e}")
            continue

    print("\n" + "=" * 80)
    print("Все эксперименты с RNN успешно завершены.")
    print("=" * 80)

if __name__ == "__main__":
    main()