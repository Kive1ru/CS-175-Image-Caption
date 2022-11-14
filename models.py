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


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim) -> None:
        super(TextEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed1 = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embed1(x)


class Decoder(nn.Module):
    def __init__(self, embed_dim, vocab_size, lstm_layer_num, EOS_token_id, max_token_num=15):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.lstm_layer_num = lstm_layer_num
        self.EOS_token_id = EOS_token_id
        self.max_token_num = max_token_num
        self.text_encoder = TextEncoder(self.vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.vocab_size, num_layers=self.lstm_layer_num)

    def forward(self, feature_vector):
        device = get_device()
        lstm_input = feature_vector.reshape((1, -1))
        token_softmax = []
        prev_token_id = None
        h_n = torch.zeros((self.lstm_layer_num, self.vocab_size)).to(device)
        c_n = torch.zeros((self.lstm_layer_num, self.vocab_size)).to(device)
        while prev_token_id != self.EOS_token_id and len(token_softmax) < self.max_token_num:
            out, (h_n, c_n) = self.lstm(lstm_input, (h_n, c_n))
            token_softmax.append(out)

            prev_token_id = out.argmax()
            lstm_input = self.text_encoder(torch.tensor([prev_token_id], dtype=int, device=device))
        return torch.concat(token_softmax, dim=0)

    def predict(self, feature_vector):
        token_softmaxs = self.forward(feature_vector)
        token_ids = token_softmaxs.argmax(dim=1)
        return token_ids


class BaselineRNN(nn.Module):
    def __init__(self, embed_dim, vocab_size, vgg_weights, EOS_token_id) -> None:
        super(BaselineRNN, self).__init__()
        self.img_encoder = VGG16ImageEncoder(weights=vgg_weights, out_size=embed_dim)
        self.decoder = Decoder(embed_dim, vocab_size, 1, EOS_token_id)

    def forward(self, x):
        features = self.img_encoder(x)
        return self.decoder(features)

    def predict(self, x):
        features = self.img_encoder(x)
        return self.decoder.predict(features)


if __name__ == "__main__":
    root = Path('data/flickr8k')
    device = get_device()

    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Read the image
    image = Image.open(root / 'Images/667626_18933d713e.jpg')

    # Convert the image to PyTorch tensor
    tensor = transform(image).to(device)
    img_cap_model = BaselineRNN(400, 2000, torchvision.models.VGG16_Weights.DEFAULT, 3).to(device)
    caption = img_cap_model.predict(tensor)  # tensor of token ids
    print(caption)
