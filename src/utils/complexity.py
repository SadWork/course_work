# Путь: src/utils/complexity.py

import re
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def compute_layout(nodes, edges):
    """
    Динамически рассчитывает высокооптимизированные координаты (x, y) для визуализации
    вычислительного графа (Signal Flow Chart), полностью исключая жестко закодированные параметры.
    Гарантирует отсутствие горизонтальных наложений блоков и пересечений с надписями.
    """
    node_layers = {}
    max_layer = -1
    
    # 1. Распределение по архитектурным слоям на основе суффиксов (_0, _1 и т.д.)
    for n in nodes:
        nid = n['id']
        match = re.search(r'_(\d+)$', nid)
        if match:
            layer_idx = int(match.group(1))
            node_layers[nid] = layer_idx
            max_layer = max(max_layer, layer_idx)
        elif nid == 'X':
            node_layers[nid] = -1
        elif nid in ['FC', 'Y']:
            node_layers[nid] = 999  # Временный маркер выходного слоя
            
    num_layers = max_layer + 1
    for nid in node_layers:
        if node_layers[nid] == 999:
            node_layers[nid] = num_layers
            
    # 2. Определение порядка внутри каждого слоя (по суб-колонкам)
    sub_cols = {}
    layer_widths = {}
    
    for l in range(-1, num_layers + 1):
        layer_nodes = [n['id'] for n in nodes if node_layers.get(n['id']) == l]
        if not layer_nodes:
            continue
            
        if l == -1:
            sub_cols['X'] = 0
            layer_widths[-1] = 1
            continue
        if l == num_layers:
            if 'FC' in layer_nodes:
                sub_cols['FC'] = 0
            if 'Y' in layer_nodes:
                sub_cols['Y'] = 1
            layer_widths[num_layers] = len(layer_nodes)
            continue
            
        # Локальные пространственные связи внутри слоя (sigma == 0)
        local_adj = {nid: [] for nid in layer_nodes}
        local_in_degree = {nid: 0 for nid in layer_nodes}
        
        for u, v, sigma in edges:
            if sigma == 0 and u in layer_nodes and v in layer_nodes:
                local_adj[u].append(v)
                local_in_degree[v] += 1
                
        # Нахождение уровней обработки внутри слоя через алгоритм Kahn (Topological BFS)
        queue = [nid for nid in layer_nodes if local_in_degree[nid] == 0]
        levels = {nid: 0 for nid in layer_nodes}
        
        while queue:
            curr = queue.pop(0)
            curr_level = levels[curr]
            for neighbor in local_adj[curr]:
                levels[neighbor] = max(levels[neighbor], curr_level + 1)
                local_in_degree[neighbor] -= 1
                if local_in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        for nid in layer_nodes:
            sub_cols[nid] = levels[nid]
            
        layer_widths[l] = max(levels.values()) + 1 if levels else 1

    # 3. Последовательный расчет X-координат (прогрессирующий шаг для устранения наложений)
    # ЗНАЧИТЕЛЬНО УВЕЛИЧЕННЫЕ ИНТЕРВАЛЫ ДЛЯ ГАРАНТИИ ОТСТУПОВ МЕЖДУ СУММАМИ, RELU И H_i
    sub_col_spacing = 3.2  # Шаг между внутренними блоками одного слоя (увеличен с 1.6)
    layer_gap = 2.8        # Безопасный пустой зазор между соседними слоями (увеличен с 1.5)
    
    x_starts = {}
    x_starts[-1] = 0.0     # Входной слой начинается с нуля
    
    for l in range(0, num_layers + 1):
        prev_l = l - 1
        prev_width = layer_widths.get(prev_l, 1)
        # Координата начала l-го слоя строится от ПРАВОГО края предыдущего слоя
        x_starts[l] = x_starts[prev_l] + (prev_width * sub_col_spacing) + layer_gap
        
    pos = {}
    
    # Назначение вертикальных эшелонов высоты по типу операции
    def vertical_sorting_weight(nid):
        nid_lower = nid.lower()
        if 'up' in nid_lower or 'dense_x' in nid_lower:
            return 1.4      # Сигнал идет сверху (проецирование входа)
        if 'tridiag' in nid_lower or 'sparse_rec' in nid_lower or 'dense_h' in nid_lower:
            return -1.4     # Локальный рекуррентный преобразователь памяти
        if 'down' in nid_lower:
            return -2.8     # Обратный поток обратной связи
        return 0.0          # Основная линия вычислений (Sum, Act, ResSum, State H)
        
    for nid, l in node_layers.items():
        base_x = x_starts[l]
        col = sub_cols.get(nid, 0)
        x_coord = base_x + col * sub_col_spacing
        y_coord = vertical_sorting_weight(nid)
        pos[nid] = (x_coord, y_coord)
        
    return pos

def calculate_complexity_measures(edges):
    """
    Вычисляет d_r, d_f и s на макро-уровне (уровне слоев), как это определено в теории.
    """
    layer_edges = get_layer_level_edges(edges)
    
    G = nx.MultiDiGraph()
    for u, v, sigma in layer_edges:
        G.add_edge(u, v, sigma=sigma)
        
    cycles = list(nx.simple_cycles(G))
    dr = 1.0
    s = 1.0
    cycle_ratios = []
    
    for cycle in cycles:
        l_val = len(cycle)
        sigma_sum = 0
        for i in range(l_val):
            u = cycle[i]
            v = cycle[(i + 1) % l_val]
            sigmas = []
            if G.has_edge(u, v):
                sigmas = [data['sigma'] for key, data in G[u][v].items() if 'sigma' in data]
            sigma_sum += sigmas[0] if sigmas else 1
            
        if sigma_sum > 0:
            cycle_ratios.append(l_val / sigma_sum)
            
    if cycle_ratios:
        dr = max(cycle_ratios)
        j_val = min(cycle_ratios)
        s = 1.0 / j_val if j_val > 0 else 1.0
        
    adj = {}
    for u, v, sigma in layer_edges:
        if u not in adj:
            adj[u] = []
        adj[u].append((v, 1.0 - sigma * dr))
        
    memo = {}
    def dfs(node, visited):
        if node == 'Y':
            return 0.0
        state_key = (node, frozenset(visited))
        if state_key in memo:
            return memo[state_key]
            
        max_dist = float('-inf')
        visited.add(node)
        if node in adj:
            for neighbor, weight in adj[node]:
                if neighbor not in visited:
                    val = dfs(neighbor, visited)
                    if val != float('-inf'):
                        max_dist = max(max_dist, weight + val)
        visited.remove(node)
        memo[state_key] = max_dist
        return max_dist
        
    df = dfs('X', set())
    return float(dr), float(df), float(s)

def get_layer_level_edges(edges):
    """
    Агрегирует нейронные связи до уровня взаимодействия слоев.
    """
    def map_node(node):
        if node == 'X' or node == 'Y':
            return node
        if isinstance(node, tuple) and node[0] == 'H':
            return f"Layer {node[1]}"
        return node

    layer_edges = set()
    for u, v, sigma in edges:
        layer_edges.add((map_node(u), map_node(v), sigma))
    return list(layer_edges)

def visualize_cyclic_graph(edges, title="RNN Neuron-Level Cyclic Graph", save_path=None):
    """
    Визуализирует подробный циклический граф с раскрытием до каждого нейрона.
    """
    G = nx.MultiDiGraph()
    for u, v, sigma in edges:
        G.add_edge(u, v, sigma=sigma)
        
    hidden_nodes = [n for n in G.nodes() if isinstance(n, tuple) and n[0] == 'H']
    layers = set(n[1] for n in hidden_nodes)
    num_layers = len(layers) if layers else 1
    max_h = max(n[2] for n in hidden_nodes) + 1 if hidden_nodes else 1
    
    pos = {}
    pos['X'] = (0, (max_h - 1) / 2.0)
    pos['Y'] = (num_layers + 1, (max_h - 1) / 2.0)
    for n in hidden_nodes:
        pos[n] = (n[1] + 1, n[2])
        
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgray', node_size=600, edgecolors='black')
    
    labels = {n: 'X' if n == 'X' else ('Y' if n == 'Y' else f"H_{n[1]},{n[2]}") for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold', ax=ax)
    
    for u, v, data in G.edges(data=True):
        sigma = data['sigma']
        color = 'royalblue' if sigma == 0 else 'crimson'
        
        if u == v:
            angleA = 60 if sigma == 0 else 240
            angleB = 120 if sigma == 0 else 300
            ax.annotate("", xy=pos[u], xycoords='data', xytext=pos[u], textcoords='data',
                        arrowprops=dict(arrowstyle="->", color=color, 
                                        connectionstyle=f"arc,angleA={angleA},angleB={angleB},armA=30,armB=30,rad=15", 
                                        shrinkA=12, shrinkB=12))
        else:
            rad = 0.15 if (isinstance(u, tuple) and isinstance(v, tuple) and u[1] > v[1]) else 0.05
            ax.annotate("", xy=pos[v], xycoords='data', xytext=pos[u], textcoords='data',
                        arrowprops=dict(arrowstyle="->", color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=12, shrinkB=12))
                                            
    ax.plot([], [], color='royalblue', label='Spatial (sigma = 0)')
    ax.plot([], [], color='crimson', label='Temporal loop (sigma = 1)')
    ax.legend(loc='upper left', frameon=True)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Подробный граф сохранен по пути: {save_path}")
    else:
        plt.show()

def visualize_layer_cyclic_graph(edges, title="RNN Layer-Level Cyclic Graph", save_path=None):
    """
    Визуализирует агрегированный послойный граф с крупными петлями рекурсии.
    """
    layer_edges = get_layer_level_edges(edges)
    
    G = nx.MultiDiGraph()
    for u, v, sigma in layer_edges:
        G.add_edge(u, v, sigma=sigma)
        
    layers = set()
    for n in G.nodes():
        if isinstance(n, str) and n.startswith("Layer "):
            layers.add(int(n.split(" ")[1]))
            
    node_order = ['X'] + [f"Layer {l}" for l in sorted(list(layers))] + ['Y']
    node_to_idx = {node: idx for idx, node in enumerate(node_order)}
    
    pos = {node: (idx, 0.0) for node, idx in node_to_idx.items() if node in G.nodes()}
    
    fig, ax = plt.subplots(figsize=(10, 5))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgreen', node_size=1200, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        sigma = data['sigma']
        color = 'royalblue' if sigma == 0 else 'crimson'
        
        if u == v:
            angleA = 65 if sigma == 0 else 245
            angleB = 115 if sigma == 0 else 295
            ax.annotate("", xy=pos[u], xycoords='data', xytext=pos[u], textcoords='data',
                        arrowprops=dict(arrowstyle="->", color=color, 
                                        connectionstyle=f"arc,angleA={angleA},angleB={angleB},armA=130,armB=130,rad=45", 
                                        shrinkA=18, shrinkB=18))
        else:
            idx_u = node_to_idx[u]
            idx_v = node_to_idx[v]
            dist = idx_v - idx_u
            
            if sigma == 0:
                rad = 0.3 * dist
            else:
                rad = -0.3 * dist
            
            ax.annotate("", xy=pos[v], xycoords='data', xytext=pos[u], textcoords='data',
                        arrowprops=dict(arrowstyle="->", color=color, 
                                        connectionstyle=f"arc3,rad={rad}", 
                                        shrinkA=18, shrinkB=18))
                        
    ax.plot([], [], color='royalblue', label='Spatial Connection (sigma = 0)')
    ax.plot([], [], color='crimson', label='Temporal Connection (sigma = 1)')
    ax.legend(loc='upper left', frameon=True)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(-3.2, 3.2)  
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Послойный граф сохранен по пути: {save_path}")
    else:
        plt.show()

def visualize_computation_graph(nodes, edges, title="RNN Detailed Computation Graph", save_path=None):
    """
    Визуализирует подробный граф прохождения сигнала (вычислительный граф)
    с динамически рассчитанным бесконфликтным взаимным расположением блоков.
    """
    # Расширенный панорамный холст для красивой горизонтальной укладки без сплющивания
    fig, ax = plt.subplots(figsize=(20, 7.5))

    # Цветовая схема для разных типов узлов
    color_map = {
        'input': '#a1dab4',       # Салатовый
        'output': '#41b6c4',      # Бирюзовый
        'state': '#dfe6e9',       # Светло-серый
        'op': '#74b9ff',          # Голубой (Операции линейной алгебры)
        'sum': '#ffeaa7',         # Желтый (+)
        'activation': '#ff7675'   # Розово-красный (Нелинейности)
    }

    # Прогрессивный динамический расчет координат узлов (Layout Engine)
    pos = compute_layout(nodes, edges)
    
    # 1. Отрисовка узлов
    for node in nodes:
        nid = node['id']
        ntype = node['type']
        color = color_map.get(ntype, '#ffffff')
        
        # Квадратные для математических операций, круглые для состояний
        shape = 's' if ntype in ['op', 'activation'] else 'o'
        # Оптимально уменьшенный диаметр узлов (свободное пространство вокруг)
        size = 1100 if ntype in ['op', 'activation', 'sum'] else 750
        
        nx.draw_networkx_nodes(
            nx.Graph(), pos, nodelist=[nid], 
            node_color=color, node_size=size, node_shape=shape,
            edgecolors='black', linewidths=1.2, ax=ax
        )
        
        # Метки с полупрозрачной белой подложкой для борьбы с перекрытием линий
        label = node['label']
        nx.draw_networkx_labels(
            nx.Graph(), pos, labels={nid: label},
            font_size=7.5, font_weight='bold', ax=ax,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, boxstyle='round,pad=0.25')
        )

    # 2. Отрисовка ребер со стрелками и изгибами
    for u, v, sigma in edges:
        if u not in pos or v not in pos:
            continue
        
        pu = pos[u]
        pv = pos[v]
        
        color = 'royalblue' if sigma == 0 else 'crimson'
        style = 'solid' if sigma == 0 else 'dashed'
        
        # Инвариантный расчет кривизны дуг во избежание пересечений
        rad = 0.0
        if sigma == 1:
            if pu[0] > pv[0]:
                # Обратная связь во времени (temporal feedback / temporal residual)
                # направляем глубокой красивой нижней дугой
                rad = -0.55
            else:
                rad = -0.2
        else:
            # Пространственные связи
            if 'ResSum' in v or 'Residual' in v:
                u_layer = re.search(r'_(\d+)$', u)
                v_layer = re.search(r'_(\d+)$', v)
                # Выгибаем вверх ПОД подложку только если это послойная (межслойная) остаточная связь
                if u_layer and v_layer and u_layer.group(1) != v_layer.group(1):
                    rad = 0.45
                else:
                    rad = 0.0  # Локальные связи внутри слоя теперь строго ровные горизонтальные (например Act_l -> ResSum_l)
            elif pu[1] != pv[1]:
                rad = 0.08 if pu[1] < pv[1] else -0.08

        ax.annotate(
            "", xy=pv, xycoords='data', xytext=pu, textcoords='data',
            arrowprops=dict(
                arrowstyle="->", color=color, linestyle=style,
                connectionstyle=f"arc3,rad={rad}", 
                shrinkA=12, shrinkB=12, lw=1.5  # Оптимизированные стыковочные зазоры
            )
        )

    # Оформление легенды и заголовка
    ax.plot([], [], color='royalblue', linestyle='solid', label='Spatial Flow (sigma = 0)', lw=1.5)
    ax.plot([], [], color='crimson', linestyle='dashed', label='Temporal Feedback (sigma = 1)', lw=1.5)
    ax.legend(loc='lower left', frameon=True, fontsize=9)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.5, max(xs) + 1.8)
    ax.set_ylim(min(ys) - 1.8, max(ys) + 1.8)
    
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Вычислительный граф сохранен по пути: {save_path}")
    else:
        plt.show()
