# Путь: src/analyze_complexity.py

import os
import argparse
from src.utils.complexity import (
    calculate_complexity_measures, 
    visualize_cyclic_graph, 
    visualize_layer_cyclic_graph,
    visualize_computation_graph
)
from src.models import StandardStackedRNN, Diagonal3TimeResModel, SparseRandomTimeResModel

def analyze_single_model(model, model_name, num_layers, hidden_size, k=None, save_dir="plots"):
    """
    Выполняет расчет характеристик и сохраняет три типа графиков.
    """
    # 1. Запрос точного циклического графа на уровне нейронов для расчета d_r, d_f, s
    edges_neuron = model.get_cyclic_graph(hidden_size)
    dr, df, s = calculate_complexity_measures(edges_neuron)
    
    # 2. Динамический запрос вычислительного графа преобразований (без хардкода)
    nodes_comp, edges_comp = model.get_computation_graph()
    
    os.makedirs(save_dir, exist_ok=True)
    base_filename = f"{model_name}_L{num_layers}_H{hidden_size}"
    if model_name == "sparse_random":
        base_filename += f"_k{k or 'auto'}"
        
    # График A: Нейронный циклический граф
    detail_path = os.path.join(save_dir, f"{base_filename}_neuron_level.png")
    title_detail = f"{model_name.upper()} Neuron-Level | d_r={dr:.2f}, d_f={df:.2f}, s={s:.2f}"
    visualize_cyclic_graph(edges_neuron, title=title_detail, save_path=detail_path)
    
    # График B: Агрегированный послойный макро-граф
    layer_path = os.path.join(save_dir, f"{base_filename}_layer_level.png")
    title_layer = f"{model_name.upper()} Layer-Level | d_r={dr:.2f}, d_f={df:.2f}, s={s:.2f}"
    visualize_layer_cyclic_graph(edges_neuron, title=title_layer, save_path=layer_path)
    
    # График C: ДЕТАЛЬНЫЙ вычислительный граф (Signal Flow Chart)
    comp_path = os.path.join(save_dir, f"{base_filename}_signal_flow.png")
    title_comp = f"{model_name.upper()} Signal Flow (Detailed Computation Diagram)"
    visualize_computation_graph(nodes_comp, edges_comp, title=title_comp, save_path=comp_path)
    
    return dr, df, s, (detail_path, layer_path, comp_path)

def main():
    parser = argparse.ArgumentParser(description="Анализ сложности архитектур RNN для курсовой работы")
    parser.add_argument("--model", type=str, default="all", 
                        choices=["standard", "diagonal", "sparse_random", "all"],
                        help="Какую модель анализировать (или 'all' для сравнения всех трех)")
    parser.add_argument("--num_layers", type=int, default=4, help="Количество слоев в архитектуре")
    parser.add_argument("--hidden_size", type=int, default=4, 
                        help="Размер скрытого состояния для построения репрезентативного графа")
    parser.add_argument("--target_diameter", type=float, default=3.0, help="Желаемый диаметр графа (для sparse_random)")
    parser.add_argument("--k", type=int, default=2, help="Количество случайных связей на узел (для sparse_random)")
    parser.add_argument("--save_dir", type=str, default="plots", help="Папка для сохранения графиков")
    
    args = parser.parse_args()
    
    models_to_analyze = ["standard", "diagonal", "sparse_random"] if args.model == "all" else [args.model]
    
    results = []
    
    print("\n" + "="*80)
    print(f"СТАРТ АНАЛИЗА СЛОЖНОСТИ (Слоев={args.num_layers}, Hidden Size={args.hidden_size})")
    print("="*80)
    
    for m in models_to_analyze:
        print(f"Инициализируем и анализируем архитектуру: {m.upper()}...")
        
        # Инстанцируем модель с переданными параметрами
        if m == "standard":
            model = StandardStackedRNN(input_size=1, hidden_size=args.hidden_size, num_layers=args.num_layers)
        elif m == "diagonal":
            model = Diagonal3TimeResModel(input_size=1, hidden_size=args.hidden_size, num_layers=args.num_layers)
        elif m == "sparse_random":
            model = SparseRandomTimeResModel(
                input_size=1,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                target_diameter=args.target_diameter,
                k=args.k
            )
            
        dr, df, s, paths = analyze_single_model(
            model=model,
            model_name=m,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            k=args.k if m == "sparse_random" else None,
            save_dir=args.save_dir
        )
        results.append((m.upper(), dr, df, s))
    
    # Сводная таблица результатов в формате Markdown
    print("\n" + "="*80)
    print("СВОДНАЯ ТАБЛИЦА СЛОЖНОСТИ ДЛЯ ОТЧЕТА")
    print("="*80)
    print(f"| {'Архитектура':<15} | {'Рекурр. глубина (d_r)':<21} | {'Прямая глубина (d_f)':<20} | {'Коэфф. пропуска (s)':<19} |")
    print(f"|{'-'*17}|{'-'*23}|{'-'*22}|{'-'*21}|")
    for name, dr, df, s in results:
        print(f"| {name:<15} | {dr:<21.2f} | {df:<20.2f} | {s:<19.2f} |")
    print("="*80)
    
    print(f"\nДетальные схемы вычислений (signal_flow.png) и графы связей сохранены в: '{args.save_dir}/'")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()