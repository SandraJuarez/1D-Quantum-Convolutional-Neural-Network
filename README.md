# 1D Quantum Convolutional Neural Network (1D-QCNN)

This repository contains the implementation of the model proposed in the following paper:

 **[Fourier-Guided Design of Quantum Convolutional Neural Networks for Time Series Forecasting](https://arxiv.org/pdf/2404.15377)**  
*Sandra Juárez Osorio, et al., 2024*

---

##  Overview

This project implements a **1D Quantum Convolutional Neural Network (1D-QCNN)** for time series forecasting. The architecture integrates quantum circuits into classical convolutional models using **quantum convolutional layers**, with the design guided by a **Fourier analysis** of the input sequences. This hybrid model leverages quantum encodings and entanglement strategies to extract high-level features from temporal data.

---

##  Related Paper

> Juárez Osorio, S., et al. (2024). Fourier-Guided Design of Quantum Convolutional Neural Networks for Time Series Forecasting. [arXiv:2404.15377](https://arxiv.org/abs/2404.15377)

If you use this codebase or find it helpful in your research, please consider citing the paper.

---

##  Dependencies

This project uses a hybrid stack combining classical and quantum machine learning libraries:

```python
import pennylane as qml
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import torch
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import os
import time

# PennyLane templates
from pennylane.templates import RandomLayers, StronglyEntanglingLayers, BasicEntanglerLayers

# Custom hybrid modules
from Quanvolution_jax2_multidim_vectorized import QuanvLayer1D
from cnn import ClassicalCNN as CNN
