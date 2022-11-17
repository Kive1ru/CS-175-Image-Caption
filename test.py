import torch
import torch.nn as nn
import nltk
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from utils import get_device

a = torch.tensor([[1,2,3], [1,2,3]])
b = a.argmax(dim=1)

print(b.transpose())
