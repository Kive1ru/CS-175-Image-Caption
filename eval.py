import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision
import torch
import matplotlib.pyplot as plt
import time
import torch.nn as nn
from PIL import Image
from pathlib import Path
from utils import get_device


def evaluate_model(model, description, pictures, tokenizer, max_cap):
    actuallist, predictlist = list(), list()
    for key, desc_list in description.items():
        prediction = ...
        actual_des = ...
        actuallist.append(actual_des)
        predictlist.append(prediction)
        
        
    print("")