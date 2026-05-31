# Путь: src/res_comparison_mlp/runner.py

import argparse
import itertools
import os
import glob
from src.datasets import get_mnist_loaders, get_fashion_mnist_loaders
from src.res_comparison_mlp.pipeline import run_training_pipeline, set_global_seeds

def parse_list_arg(arg_str, target_type):
    return [target_type(x.strip()) for x in arg_str.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser(description="Автоматический запуск сетки экспериментов ResNet MLP")
    parser.add_argument("--models", type=str, default="no_skip,standard,random_skip",
                        help="Типы блоков через запятую (no_skip, standard, random_skip)")
    parser.add_argument("--depths", type=str, default="5,10,20,50,100",
                        help="Список глубин блоков через запятую")
    parser.add_argument("--hidden_sizes", type=str, default="128",
                        help="Список ширин скрытых слоев через запятую")
    parser.add_argument("--datasets", type=str, default="mnist,fashion_mnist",
                        help="Список датасетов через запятую (mnist, fashion_mnist)")
    parser.add_argument("--epochs", type=int, default=15, help="Количество эпох на запуск")
    parser.add_argument("--lr", type=float, default=0.001, help="Скорость обучения")
    parser.add_argument("--batch_size", type=int, default=128, help="Размер батча")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно (фиксирует воспроизводимость)")
    parser.add_argument("--save_dir", type=str, default="results_pkl", help="Папка для сохранения бинарных чекпоинтов")
    args = parser.parse_args()

    # Сброс генератора для воспроизводимой загрузки данных на старте
    set_global_seeds(args.seed)

    model_list = parse_list_arg(args.models, str)
    depth_list = parse_list_arg(args.depths, int)
    hidden_size_list = parse_list_arg(args.hidden_sizes, int)
    dataset_list = parse_list_arg(args.datasets, str)

    experiments = list(itertools.product(dataset_list, model_list, depth_list, hidden_size_list))
    
    print("=" * 70)
    print(f"Запланировано экспериментов: {len(experiments)}")
    print(f"Модели:  {model_list}")
    print(f"Глубины: {depth_list}")
    print(f"Ширины:  {hidden_size_list}")
    print(f"Данные:  {dataset_list}")
    print(f"Сид:     {args.seed}")
    print("=" * 70)

    loaders_cache = {}

    for idx, (dataset_name, model_type, depth, hidden_size) in enumerate(experiments):
        print(f"\n[Эксперимент {idx + 1}/{len(experiments)}]")
        
        # Шаблон имени файла: {dataset_name}_{model_type}_L{depth}_W{hidden_size}_scale{opt_scale:.4f}_seed{seed}.pkl
        pattern = f"{dataset_name}_{model_type}_L{depth}_W{hidden_size}_scale*_seed{args.seed}.pkl"
        full_pattern = os.path.join(args.save_dir, pattern)
        
        # Если файл, соответствующий шаблону, уже существует, пропускаем эксперимент
        if glob.glob(full_pattern):
            print(f"Пропуск: Эксперимент {model_type} L{depth} W{hidden_size} на {dataset_name} уже выполнен.")
            continue
        
        if dataset_name not in loaders_cache:
            print(f"Загрузка датасета: {dataset_name.upper()}...")
            # При создании DataLoader PyTorch привязывается к текущему глобальному сиду
            set_global_seeds(args.seed)
            if dataset_name == "mnist":
                loaders_cache[dataset_name] = get_mnist_loaders(batch_size=args.batch_size)
            elif dataset_name == "fashion_mnist":
                loaders_cache[dataset_name] = get_fashion_mnist_loaders(batch_size=args.batch_size)
            else:
                print(f"Неизвестный датасет '{dataset_name}'. Пропуск конфигурации.")
                continue
                
        train_loader, test_loader = loaders_cache[dataset_name]

        try:
            run_training_pipeline(
                model_type=model_type,
                depth=depth,
                hidden_size=hidden_size,
                dataset_name=dataset_name,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                seed=args.seed,
                save_dir=args.save_dir
            )
        except Exception as e:
            print(f"Ошибка при выполнении конфигурации {model_type} L{depth} W{hidden_size}: {e}")
            continue

    print("\n" + "=" * 70)
    print("Все запланированные эксперименты завершены.")
    print("=" * 70)

if __name__ == "__main__":
    main()