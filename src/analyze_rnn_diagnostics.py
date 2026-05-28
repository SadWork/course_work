# Путь: src/analyze_rnn_diagnostics.py

import os
import pickle
import argparse
import numpy as np

def load_pkl_files(directory):
    payloads = []
    if not os.path.exists(directory):
        print(f"Директория {directory} не найдена.")
        return payloads
        
    for file in os.listdir(directory):
        if file.endswith(".pkl") :#and ("temporal" in file or "alternating" in file):
            path = os.path.join(directory, file)
            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
                    payloads.append((file, payload))
            except Exception as e:
                print(f"Ошибка при загрузке {file}: {e}")
    return payloads


def analyze_diagnostics(payloads):
    if not payloads:
        print("Нет подходящих файлов для анализа.")
        return

    print("=" * 110)
    print("ГЛУБОКИЙ АНАЛИЗ ГРАДИЕНТНОГО ПОТОКА ПО КООРДИНАТАМ (СЛОЙ, ВРЕМЯ)")
    print("=" * 110)

    for filename, data in payloads:
        meta = data.get("metadata", {})
        history = data.get("history", {})
        
        model_type = meta.get("model_type", "unknown").upper()
        depth = meta.get("depth", 0)
        seq_len = meta.get("seq_len", 0)
        dataset = meta.get("dataset_name", "unknown").upper()
        
        final_train_loss = history.get("train_loss", [-1])[-1]
        
        print(f"\nФайл: {filename}")
        print(f"Конфигурация: {model_type} | L={depth} | T={seq_len} | Train Loss={final_train_loss:.4f}")
        
        # Считываем новые координатные срезы градиентов
        state_grads_init = data.get("state_grads_init", None)
        state_grads_final = data.get("state_grads_final", None)
        
        if state_grads_init is None:
            print("  [Предупреждение] В файле отсутствуют срезы state_grads_init. Перезапустите эксперимент с новыми логами.")
            continue
            
        for name, matrix in [("Инициализация", state_grads_init), ("Конец обучения", state_grads_final)]:
            if matrix is None:
                continue
            print(f"  --- Профиль градиентов по состояниям ({name}) ---")
            print(f"  Формат: Средняя норма dL/dh_t^l")
            
            # Определяем опорные временные точки для отображения
            if seq_len <= 8:
                time_indices = list(range(seq_len))
            else:
                # Показываем начало, середину и конец последовательности
                time_indices = [
                    0, 
                    int(seq_len * 0.1), 
                    int(seq_len * 0.3), 
                    int(seq_len * 0.5), 
                    int(seq_len * 0.7), 
                    int(seq_len * 0.9), 
                    seq_len - 1
                ]
            
            # Строим текстовую таблицу-теплокарту
            header = "          " + " | ".join([f"t={t:03d}" for t in time_indices])
            print(header)
            print("-" * len(header))
            
            for l in range(depth):
                row_vals = []
                for t in time_indices:
                    val = matrix[l, t]
                    row_vals.append(f"{val:.2e}")
                print(f"  Слой {l:02d} | " + " | ".join(row_vals))
                
            # Количественная оценка затухания/взрыва по времени для каждого слоя
            print("  Анализ затухания градиента во времени (t=0 относительно t=T-1):")
            for l in range(depth):
                grad_start = matrix[l, 0]
                grad_end = matrix[l, seq_len - 1]
                ratio_time = grad_start / (grad_end + 1e-12)
                
                if ratio_time < 1e-4:
                    status = "ИСЧЕЗНОВЕНИЕ (Vanishing)"
                elif ratio_time > 1e4:
                    status = "ВЗРЫВ (Exploding)"
                else:
                    status = "Стабильное поведение"
                    
                print(f"    Слой {l:02d}: dL/dh_0 / dL/dh_T = {ratio_time:.2e} -> {status}")
                
            # Оценка затухания по глубине (Слой 0 относительно последнего слоя)
            print("  Анализ затухания градиента по глубине (Слой 0 относительно верхнего Слоя L-1):")
            for t_idx in [0, seq_len - 1]:
                grad_bottom = matrix[0, t_idx]
                grad_top = matrix[depth - 1, t_idx]
                ratio_depth = grad_bottom / (grad_top + 1e-12)
                print(f"    Шаг t={t_idx:03d}: dL/dh^{{l=0}} / dL/dh^{{l=L-1}} = {ratio_depth:.2e}")
                
        print("-" * 110)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Диагностика логов RNN")
    parser.add_argument("--dir", type=str, default="results_rnn_pkl", help="Путь к папке с результатами (.pkl)")
    args = parser.parse_args()
    
    payloads = load_pkl_files(args.dir)
    analyze_diagnostics(payloads)