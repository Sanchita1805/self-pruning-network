# Self-Pruning Neural Network

## Overview
This project implements a neural network that learns to prune its own weights during training using learnable gates and L1 sparsity regularization.

## Features
- Custom PrunableLinear layer
- Learnable gating mechanism
- L1 sparsity loss
- CIFAR-10 training
- Sparsity vs Accuracy analysis

## Results
| Lambda | Accuracy | Sparsity |
|--------|----------|----------|
| 1e-5   | 55%      | 0.28%    |
| 1e-4   | 53%      | 1.47%    |
| 1e-3   | 51%      | 1.70%    |

## Run
```bash]
python main.py
