# filepath: src/train_mnist.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import get_mnist_loaders
from models import StandardStackedRNN, Diagonal3TimeResModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_classification(model, train_loader, test_loader, device, epochs=3, lr=0.001, perm_idx=None, seq_len=784):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, seq_len, 1)
            if perm_idx is not None:
                data = data[:, perm_idx, :]

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Accuracy: {100. * correct / total:.2f}%")

        test_acc = evaluate_classification(model, test_loader, device, perm_idx, seq_len)
        train_acc = 100. * correct / total
        print(f"=== Epoch {epoch+1} Complete. Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}% ===\n")

def evaluate_classification(model, test_loader, device, perm_idx=None, seq_len=784):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, seq_len, 1)
            if perm_idx is not None:
                data = data[:, perm_idx, :]
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return 100. * correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNNs on psMNIST")
    parser.add_argument("--model", type=str, default="diagonal", choices=["standard", "diagonal"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=191)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--compile", action="store_true", help="Compile proposed model layers")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)

    seq_len = 784
    torch.manual_seed(42)
    pixel_permutation = torch.randperm(seq_len)

    if args.model == "standard":
        model = StandardStackedRNN(input_size=1, hidden_size=128, num_layers=4, output_size=10)
    else:
        model = Diagonal3TimeResModel(input_size=1, hidden_size=args.hidden_size, num_layers=args.num_layers, output_size=10)
        if args.compile:
            for i in range(len(model.layers)):
                model.layers[i] = torch.compile(model.layers[i])

    model = model.to(device)
    print(f"Model: {args.model.upper()}")
    print(f"Parameters: {count_parameters(model)}")

    train_classification(model, train_loader, test_loader, device, epochs=args.epochs, perm_idx=pixel_permutation, seq_len=seq_len)