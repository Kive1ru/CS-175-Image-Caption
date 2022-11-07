import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from utils import get_device


class VGG16ImageEncoder(nn.Module):
    def __init__(self, weights, out_size):
        super(VGG16ImageEncoder, self).__init__()
        self.out_size = out_size
        self.model = torchvision.models.vgg16(weights=weights)

        # modify the last predict layer to output the desired dimension
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, out_size)

    def freeze_param(self):
        for param in self.model.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.model.features(x)
        avgpool = self.model.avgpool(features).flatten()
        out = self.model.classifier(avgpool)
        return out


class BaselineRNN(nn.Module):
    def __init__(self, img_shape):
        pass

if __name__ == "__main__":
    root = Path('data/flickr8k')
    device = get_device()

    vgg16 = VGG16ImageEncoder(torchvision.models.VGG16_Weights.DEFAULT, 300)

    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Read the image
    image = Image.open(root/'Images/667626_18933d713e.jpg')

    # Convert the image to PyTorch tensor
    tensor = transform(image).to(device)
    vgg16.to(device)

    out = vgg16(tensor)
    print(out.shape)
