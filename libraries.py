"""
Makemore Part 3: Activations, Gradients, and Batch Normalization

Topics covered:
- Why deep networks are fragile (saturated activations, vanishing gradients)
- Proper weight initialization (Kaiming init)
- Batch Normalization for training stability
- Visualizing activations and gradients during training
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import os
