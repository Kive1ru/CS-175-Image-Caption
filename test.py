import torch
import torch.nn as nn
import nltk
import numpy as np
# import tensorflow as tf
from pathlib import Path


root = Path('data')
count = 0
for img in (root / 'images').iterdir():
    count += 1
print(count)


class BaselineRNN(nn.module):
    def __init__(self, img_shape):
        pass


    
