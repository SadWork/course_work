# filepath: src/train_adding.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import get_adding_data
from models import StandardStackedRNN, Diagonal3TimeResModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_adding(model, n_iters=2000, batch_size=64, seq_len=50, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []

    for i in range(n_iters):
        inputs, targets = get_adding_data(batch_size, seq_len)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        if i % 100 == 0:
            print(f"Iter {i}, Loss: {loss.item():.4f}")
    return losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNNs on the Adding Problem")
    parser.add_argument("--model", type=str, default="diagonal", choices=["standard", "diagonal"])
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--iters", type=int, default=2000)
    args = parser.parse_args()

    if args.model == "standard":
        model = StandardStackedRNN(input_size=2, hidden_size=60, num_layers=args.num_layers)
    else:
        model = Diagonal3TimeResModel(input_size=2, hidden_size=args.hidden_size, num_layers=args.num_layers)

    print(f"Model: {args.model.upper()}")
    print(f"Parameters: {count_parameters(model)}")
    print(f"Sequence Length: {args.seq_len}")
    train_adding(model, n_iters=args.iters, seq_len=args.seq_len)