import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from utils import get_device
import math


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
        resnet = torchvision.models.resnet50(weights=weights)  # pretrained=True

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, out_size)

    def forward(self, images):
        features = self.resnet(images)  # (batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)  # (batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1))  # (batch_size,49,2048)
        features = self.fc(features)  # (batch_size,49,out_size)
        return features

    def freeze_param(self):
        for param in self.resnet.parameters():
            param.requires_grad = False


# from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# inspired by https://towardsdatascience.com/image-captions-with-attention-in-tensorflow-step-by-step-927dad3569fa
class Img2Cap(nn.Module):
    def __init__(self, tokenizer, embed_dim, ing_encoder_weights):
        super(Img2Cap, self).__init__()
        self.d_model = embed_dim
        self.tokenizer = tokenizer
        self.embed_dim = embed_dim
        self.vocab_size = len(tokenizer.word_index) if tokenizer.num_words is None else tokenizer.num_words + 1
        self.img_encoder = ResNetEncoder(ing_encoder_weights, embed_dim)
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=5, batch_first=True)  # embed_dim must be divisible by num_heads
        self.embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx=tokenizer.word_index["<PAD>"])
        self.positional_encoding = PositionalEncoding(d_model=self.d_model)
        self.fc = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, imgs, captions):
        """
        Parameters
        ----------
        imgs: torch.Tensor
            shape: [batch_size, channel, H, W]
        captions: torch.Tensor
            shape: [batch_size, seq_len]
        src_pad_mask: torch.BoolTensor | None
            shape: [batch_size, seq_len]
        tgt_pad_mask: torch.BoolTensor | None
            shape: [batch_size, seq_len]
        """
        device = get_device()
        b_size, cap_len = captions.shape[0], captions.shape[1]
        src = self.img_encoder(imgs)  # [batch_size, "channel", embed_dim]
        # TODO: is positional encoding for src necessary?
        src = self.positional_encoding(src.transpose(0, 1) * math.sqrt(self.d_model)).transpose(0, 1)
        tgt = self.positional_encoding(self.embedding(captions).transpose(0, 1) * math.sqrt(self.d_model)).transpose(0, 1)
        tgt_mask = torch.stack([self.transformer.generate_square_subsequent_mask(cap_len) for _ in range(b_size * self.transformer.nhead)], dim=0).to(device)
        tgt_pad_mask = (captions == self.tokenizer.word_index["<PAD>"])  # a BoolTensor where True positions are not attended

        out = self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.fc(out)
        return out

    def predict(self, img, max_token_num=40):
        """
        Parameters
        ----------
        img: torch.Tensor
            shape [1, 3, H, W]
        max_token_num: int
            the maximum number of generated tokens for a sentence.

        Returns
        -------
        torch.Tensor
            1D tensor that stores token ids of the predicted sentence.
        """
        device = get_device()

        token_ids = [self.tokenizer.word_index['<SOS>']]

        src = self.img_encoder(img)  # [batch_size=1, "channel", embed_dim]
        # TODO: is positional encoding for src necessary?
        src = self.positional_encoding(src.transpose(0, 1) * math.sqrt(self.d_model)).transpose(0, 1).squeeze(dim=0)
        tgt_input = torch.Tensor(token_ids).to(dtype=int, device=get_device())
        tgt_input = self.positional_encoding(self.embedding(tgt_input).unsqueeze(1) * math.sqrt(self.d_model)).squeeze(dim=1)
        for w in range(max_token_num):
            out = self.transformer(src, tgt_input)
            out = self.fc(out)
            tgt_output = out.argmax(dim=1)

            next_token_id = tgt_output[-1].item()
            token_ids.append(next_token_id)

            if next_token_id == self.tokenizer.texts_to_sequences(["<EOS>"])[0][0]:
                break

            tgt_input = torch.Tensor(token_ids).to(dtype=int, device=device)
            tgt_input = self.positional_encoding(self.embedding(tgt_input).unsqueeze(1) * math.sqrt(self.d_model)).squeeze(dim=1)
        return token_ids[1:]  # [cap_len]


class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_layers, tokenizer, feature_dim):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_index)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.EOS_token = tokenizer.word_index['<EOS>']
        self.text_encoder = nn.Embedding(self.vocab_size, embed_dim, tokenizer.word_index['<PAD>'])
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
        h = h.reshape((b_size, self.hidden_size, self.num_layers)).permute(2, 0, 1).contiguous()
        c = c.reshape((b_size, self.hidden_size, self.num_layers)).permute(2, 0, 1).contiguous()
        return h, c

    def predict(self, features, max_token_num=35) -> list:
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
    root = Path('../data/flickr8k')
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
