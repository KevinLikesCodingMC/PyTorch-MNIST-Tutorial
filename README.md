# PyTorch-MNIST-Tutorial

## PyTorch MNIST 手写数字识别教程

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 1.9+](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/KevinLikesCodingMC/PyTorch-MNIST-Tutorial.svg?style=social)](https://github.com/KevinLikesCodingMC/PyTorch-MNIST-Tutorial)

# Quick Start

## 1. Clone the repositories

```bash
git clone https://github.com/KevinLikesCodingMC/PyTorch-MNIST-Tutorial.git
```
## 2. Install requirements

First download the [MNIST dataset in CSV format](https://github.com/phoebetronic/mnist) and place `mnist_train.csv` and `mnist_test.csv` in the root directory:

```
PyTorch-MNIST-Tutorial/
├── LICENSE
├── model.pth
├── model.py
├── README.md
├── test.py
├── train.py
├── utils.py
├── mnist_train.csv
└── mnist_test.csv
```

Then install these packages:

```bash
pip install numpy, matplotlib
```

## 3. Install PyTorch 

Visit [PyTorch official website](https://pytorch.org/get-started/locally/).

## 4. Training and Testing

Run `train.py` to train and generate model `model.pth`.

Run `test.py` to test the model `model.pth`.
