import torch
import torch.nn as nn
import torch.nn.functional as F

class Diagonal3TimeResLayer(nn.Module):
    def __init__(self, input_size: int, n_curr: int, n_next: int = None, is_last: bool = False):
        super().__init__()
        self.is_last = is_last
        self.n_curr = n_curr

        # W_up выносится наружу для предварительного расчета
        self.w_up = nn.Linear(input_size, n_curr)

        # W_time: трехдиагональная матрица
        self.diag_main = nn.Parameter(torch.Tensor(n_curr))
        self.diag_up = nn.Parameter(torch.Tensor(n_curr - 1))
        self.diag_down = nn.Parameter(torch.Tensor(n_curr - 1))

        # W_down: связь от верхнего слоя
        if not is_last:
            self.w_down = nn.Linear(n_next, n_curr)

        self.bias = nn.Parameter(torch.Tensor(n_curr))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_up.weight, a=0.1)
        if not self.is_last:
            nn.init.kaiming_uniform_(self.w_down.weight, a=0.1)

        nn.init.constant_(self.diag_main, 0.0)
        nn.init.constant_(self.diag_up, 0.0)
        nn.init.constant_(self.diag_down, 0.0)
        nn.init.constant_(self.bias, 0)

    def forward(self, x_proj_t: torch.Tensor, h_prev: torch.Tensor, h_next_layer_prev: torch.Tensor = None) -> torch.Tensor:
        # 1. Вычисляем циклическую часть без in-place модификаций
        main_term = h_prev * self.diag_main

        # Сдвиги через конкатенацию (эффективно для компилятора)
        up_term = torch.cat([
            h_prev[:, 1:] * self.diag_up,
            torch.zeros(h_prev.size(0), 1, device=h_prev.device)
        ], dim=1)

        down_term = torch.cat([
            torch.zeros(h_prev.size(0), 1, device=h_prev.device),
            h_prev[:, :-1] * self.diag_down
        ], dim=1)

        res_time = main_term + up_term + down_term

        # 2. Собираем сигналы (используем уже готовый спроецированный x_proj_t)
        res = x_proj_t + res_time + self.bias
        if not self.is_last and h_next_layer_prev is not None:
            res = res + self.w_down(h_next_layer_prev)

        # 3. Residual шаг
        delta = F.leaky_relu(res, negative_slope=0.1)
        return h_prev + delta


class Diagonal3TimeResModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            self.layers.append(Diagonal3TimeResLayer(input_size, hidden_size, hidden_size, is_last))

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # Считаем проекцию входа один раз для всей последовательности
        x_projected = [self.layers[l].w_up(x) for l in range(self.num_layers)]

        # Инициализация состояний
        h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        for t in range(seq_len):
            new_h = []
            for l in range(self.num_layers):
                h_next = h[l+1] if l + 1 < self.num_layers else None
                # Передаем предрассчитанный шаг x
                h_curr_new = self.layers[l](x_projected[l][:, t, :], h[l], h_next)
                new_h.append(h_curr_new)
            h = new_h

        return self.fc(h[-1])