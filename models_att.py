import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nn_f
import torchvision
import torchvision.transforms as transforms

import einops

from PIL import Image
from pathlib import Path
from utils import get_device


class VGG16ImageEncoder(nn.Module):
    def __init__(self, weights, out_size):
        super(VGG16ImageEncoder, self).__init__()
        self.out_size = out_size
        model = torchvision.models.resnet50(weights=weights)

        # modify the last predict layer to output the desired dimension
        '''self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(25088),
            nn.Linear(25088, 10000),
            nn.Tanh(),
            nn.Linear(10000, out_size),
            nn.Tanh()
        )'''
        for param in model.parameters():
            param.requires_grad = False
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)


    '''def freeze_param(self):
        for param in self.model.features.parameters():
            param.requires_grad = False'''

    def forward(self, x):
        #out = self.model.features(x).reshape((x.shape[0], -1))
        features = self.model(x)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))
        return features


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx) -> None:
        super(TextEncoder, self).__init__()
        self.embed1 = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

    def forward(self, x):
        return self.embed1(x)

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim,attention_dim)
        self.decoder_att = nn.Linear(decoder_dim,attention_dim)
        self.full_att = nn.Linear(attention_dim,1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward (self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha

class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_size, tokenizer, feature_dim):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_index)
        self.EOS_token = tokenizer.texts_to_sequences(["endseq"])[0][0]
        self.text_encoder = TextEncoder(self.vocab_size, embed_dim, tokenizer.texts_to_sequences(["<PAD>"])[0][0])
        self.init_h = nn.Linear(feature_dim, hidden_size)
        self.init_c = nn.Linear(feature_dim, hidden_size)
        self.lstm_cell = nn.LSTMCell(embed_dim, hidden_size, bias=True)
        '''self.fc = nn.Sequential(
            nn.Linear(hidden_size, self.vocab_size, bias=True),
            nn.Sigmoid()
        )'''
        self.fc = nn.Linear(hidden_size, self.vocab_size, bias=True)
        self.drop = nn.Dropout(0.03)

    def forward(self, features, captions):
        b_size = captions.shape[0]
        token_softmaxs = []
        h, c = self.init_hidden(features)
        embeddings = self.text_encoder(captions)
        for w in range(captions.shape[1]):
            h, c = self.lstm_cell(embeddings[:, w, :], (h, c))
            out = self.fc(self.drop(h))
            token_softmaxs.append(out)

        return torch.stack(token_softmaxs, dim=0).to(get_device())

    def init_hidden(self, features):
        return self.init_h(features.mean(dim=1)), self.init_c(features.mean(dim=1))

    # def predict(self, features, max_token_num=35):
    #     b_size = features.shape[0]
    #     token_softmaxs = []  # token_softmaxs = torch.zeros((captions.shape[1], b_size, self.vocab_size)).to(get_device())
    #     h, c = self.init_hidden(features)
    #     SOS_token = torch.Tensor(self.tokenizer.texts_to_sequences([["startseq" for _ in range(b_size)]])[0]).to(dtype=int, device=get_device())
    #     prev_embeddings = self.text_encoder(SOS_token)
    #     sentences = None
    #
    #     for w in range(max_token_num):
    #         h, c = self.lstm_cell(prev_embeddings, (h, c))
    #         out = self.fc(h)
    #         new_words = out.argmax(dim=1)
    #         prev_embeddings = self.text_encoder(new_words)
    #         new_words = new_words.reshape((out.shape[0], 1))
    #         if sentences is None:
    #             sentences = new_words
    #         else:
    #             sentences = torch.cat((sentences, new_words), dim=1)
    #
    #         # if every sentence has reached end of sentence, break
    #         if (sentences == self.EOS_token).any(dim=1).to(torch.float32).mean() == 1:
    #             break
    #     return sentences

    def predict(self, features, max_token_num=35):
        h, c = self.init_hidden(features)
        SOS_token = torch.Tensor(self.tokenizer.texts_to_sequences([["startseq"]])[0]).to(dtype=int, device=get_device())
        prev_embeddings = self.text_encoder(SOS_token)
        sentence = []

        for w in range(max_token_num):
            h, c = self.lstm_cell(prev_embeddings, (h, c))
            out = self.fc(h)
            new_word = out.argmax(dim=1)
            sentence.append(new_word.item())

            # if every sentence has reached end of sentence, break
            if new_word.item() == self.EOS_token:
                break
            prev_embeddings = self.text_encoder(new_word)
        return sentence


class Decoder_attension(nn.Module):
    def __init__(self, embed_dim, hidden_size, tokenizer, feature_dim):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_index)
        self.EOS_token = tokenizer.texts_to_sequences(["endseq"])[0][0]
        self.text_encoder = TextEncoder(self.vocab_size, embed_dim, tokenizer.texts_to_sequences(["<PAD>"])[0][0])
        self.init_h = nn.Linear(feature_dim, hidden_size)
        self.init_c = nn.Linear(feature_dim, hidden_size)
        self.lstm_cell = nn.LSTMCell(embed_dim, hidden_size, bias=True)
        '''self.fc = nn.Sequential(
            nn.Linear(hidden_size, self.vocab_size, bias=True),
            nn.Sigmoid()
        )'''
        self.fc = nn.Linear(hidden_size, self.vocab_size, bias=True)
        self.drop = nn.Dropout(0.03)

    def forward(self, features, captions):
        b_size = captions.shape[0]
        token_softmaxs = []
        h, c = self.init_hidden(features)
        embeddings = self.text_encoder(captions)
        for w in range(captions.shape[1]):
            h, c = self.lstm_cell(embeddings[:, w, :], (h, c))
            out = self.fc(self.drop(h))
            token_softmaxs.append(out)

        return torch.stack(token_softmaxs, dim=0).to(get_device())

    def init_hidden(self, features):
        return self.init_h(features.mean(dim=1)), self.init_c(features.mean(dim=1))

    # def predict(self, features, max_token_num=35):
    #     b_size = features.shape[0]
    #     token_softmaxs = []  # token_softmaxs = torch.zeros((captions.shape[1], b_size, self.vocab_size)).to(get_device())
    #     h, c = self.init_hidden(features)
    #     SOS_token = torch.Tensor(self.tokenizer.texts_to_sequences([["startseq" for _ in range(b_size)]])[0]).to(dtype=int, device=get_device())
    #     prev_embeddings = self.text_encoder(SOS_token)
    #     sentences = None
    #
    #     for w in range(max_token_num):
    #         h, c = self.lstm_cell(prev_embeddings, (h, c))
    #         out = self.fc(h)
    #         new_words = out.argmax(dim=1)
    #         prev_embeddings = self.text_encoder(new_words)
    #         new_words = new_words.reshape((out.shape[0], 1))
    #         if sentences is None:
    #             sentences = new_words
    #         else:
    #             sentences = torch.cat((sentences, new_words), dim=1)
    #
    #         # if every sentence has reached end of sentence, break
    #         if (sentences == self.EOS_token).any(dim=1).to(torch.float32).mean() == 1:
    #             break
    #     return sentences

    def predict(self, features, max_token_num=35):
        h, c = self.init_hidden(features)
        SOS_token = torch.Tensor(self.tokenizer.texts_to_sequences([["startseq"]])[0]).to(dtype=int, device=get_device())
        prev_embeddings = self.text_encoder(SOS_token)
        sentence = []

        for w in range(max_token_num):
            h, c = self.lstm_cell(prev_embeddings, (h, c))
            out = self.fc(h)
            new_word = out.argmax(dim=1)
            sentence.append(new_word.item())

            # if every sentence has reached end of sentence, break
            if new_word.item() == self.EOS_token:
                break
            prev_embeddings = self.text_encoder(new_word)
        return sentence

class BaselineRNN(nn.Module):
    def __init__(self, embed_dim, hidden_size, tokenizer, feature_dim, vgg_weights) -> None:
        super(BaselineRNN, self).__init__()
        self.img_encoder = VGG16ImageEncoder(weights=vgg_weights, out_size=feature_dim)
        self.decoder = Decoder(embed_dim, hidden_size, tokenizer, feature_dim)

    def forward(self, x, captions):
        features = self.img_encoder(x)
        return self.decoder(features, captions)

    def predict(self, x):
        features = self.img_encoder(x)
        return self.decoder.predict(features)


if __name__ == "__main__":
    root = Path('data/flickr8k')
    #device = get_device()
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Read the image
    image = Image.open(root / 'Images/667626_18933d713e.jpg')

    # Convert the image to PyTorch tensor
    #tensor = transform(image).to(device)
    # img_cap_model = BaselineRNN(400, 2000, torchvision.models.VGG16_Weights.DEFAULT, 3).to(device)

    vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    print(vgg)

    net50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    print(net50)
    # model = BaselineRNN(400, 8496, 3000, torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES, 3, 1)
    # print(model)
