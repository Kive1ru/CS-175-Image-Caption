import torch
import torch.nn as nn
import nltk
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from utils import get_device


root = Path('data/flickr8k')
count = 0
for img in (root / 'images').iterdir():
    count += 1
print(count)

device = get_device()
print(device)
