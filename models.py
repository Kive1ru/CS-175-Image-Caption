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
        # nn.Sequential(
        #     nn.Linear(self.model.classifier[6].in_features, out_size),
        #     nn.Tanh()
        # )

    def freeze_param(self):
        for param in self.model.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.model(x).reshape((x.shape[0], -1))
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, weights, out_size):
        super(ResNetEncoder, self).__init__()
        self.out_size = out_size
        self.model = torchvision.models.resnet50(weights=weights)  # pretrained=True

        # modify the last predict layer to output the desired dimension
        self.model.fc = nn.Linear(self.model.fc.in_features, out_size)

    def forward(self, x):
        return self.model(x)

    def freeze_param(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx) -> None:
        super(TextEncoder, self).__init__()
        self.embed1 = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

    def forward(self, x):
        return self.embed1(x)


class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_layers, tokenizer, feature_dim):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_index)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.EOS_token = tokenizer.word_index['<EOS>']
        self.text_encoder = TextEncoder(self.vocab_size, embed_dim, tokenizer.word_index['<PAD>'])
        self.init_h = nn.Linear(feature_dim, hidden_size * num_layers)
        self.init_c = nn.Linear(feature_dim, hidden_size * num_layers)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, self.vocab_size, bias=True)
        )

    def forward(self, features, captions):
        b_size = captions.shape[0]
        token_softmaxs = []
        h, c = self.init_hidden(features)
        embeddings = self.text_encoder(captions)
        for w in range(captions.shape[1]):
            input = embeddings[:, w, :].reshape((1, b_size, self.embed_dim))
            out, (h, c) = self.lstm(input, (h, c))
            out = self.fc(out[0])
            token_softmaxs.append(out)

        return torch.stack(token_softmaxs, dim=0).to(get_device())

    def init_hidden(self, features):
        b_size = features.shape[0]
        h, c = self.init_h(features), self.init_c(features)
        h = h.reshape((b_size, self.hidden_size, self.num_layers)).transpose(0, 2).transpose(1, 2)
        c = c.reshape((b_size, self.hidden_size, self.num_layers)).transpose(0, 2).transpose(1, 2)
        return h, c

    def predict(self, features, max_token_num=35):
        h, c = self.init_hidden(features)
        SOS_token = torch.Tensor([self.tokenizer.word_index['<SOS>']]).to(dtype=int, device=get_device())
        prev_embeddings = self.text_encoder(SOS_token)
        sentence = []

        for w in range(max_token_num):
            out, (h, c) = self.lstm(prev_embeddings.reshape((1, 1, self.embed_dim)), (h, c))
            out = self.fc(out[0])
            new_word = out.argmax(dim=1)
            sentence.append(new_word.item())

            # if every sentence has reached end of sentence, break
            if new_word.item() == self.EOS_token:
                break
            prev_embeddings = self.text_encoder(new_word)
        return sentence


class BaselineRNN(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_layers, tokenizer, feature_dim, weights) -> None:
        super(BaselineRNN, self).__init__()
        self.img_encoder = ResNetEncoder(weights=weights, out_size=feature_dim)
        self.decoder = Decoder(embed_dim, hidden_size, num_layers, tokenizer, feature_dim)

    def forward(self, x, captions):
        features = self.img_encoder(x)
        return self.decoder(features, captions)

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
    # img_cap_model = BaselineRNN(400, 2000, torchvision.models.VGG16_Weights.DEFAULT, 3).to(device)

    vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    # print(vgg)

    net50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    for c in list(net50.children())[:-2]:
        print(c)
    # model = BaselineRNN(400, 8496, 3000, torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES, 3, 1)
    # print(model)
