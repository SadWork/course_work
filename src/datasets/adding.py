import torch
import numpy as np

def get_adding_data(batch_size: int, seq_len: int):
    # Значения от 0 до 1
    values = torch.rand(batch_size, seq_len, 1)
    # Маска: в каждой строке ровно две единицы
    masks = torch.zeros(batch_size, seq_len, 1)
    for i in range(batch_size):
        indices = np.random.choice(seq_len, 2, replace=False)
        masks[i, indices, 0] = 1.0

    inputs = torch.cat([values, masks], dim=2)
    targets = (values * masks).sum(dim=1)
    return inputs, targets