# Путь: src/datasets/__init__.py

from .adding import get_adding_data
from .mnist import get_mnist_loaders, get_fashion_mnist_loaders

__all__ = ["get_adding_data", "get_mnist_loaders", "get_fashion_mnist_loaders"]