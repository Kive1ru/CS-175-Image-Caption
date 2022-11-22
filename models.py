import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from utils import get_device
import tensorflow as tf
import keras
from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu

# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from keras.utils import plot_model


from sklearn.utils import shuffle

# class LocAttention(tf.keras.model):
#     def __init__(self,units):
#         self.W1 = tf.keras.layers.Dense(units)
#         self.W2 = tf.keras.layers.Dense(units)

class VGG16Encoder(tf.keras.Model):
    def __init__(self,embedding_dim):
        super(VGG16Encoder,self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)
        self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None,seed=None)
    
    def call(self,x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x
        
class AttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=True):
        super(AttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
    def forward(self, l, g):
        N, C, W, H = l.size()
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        if self.up_factor > 1:
            g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        c = self.phi(F.relu(l_ + g_)) # batch_sizex1xWxH
        
        # compute attn map
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l) # batch_sizexCxWxH
        if self.normalize_attn:
            output = f.view(N,C,-1).sum(dim=2) # weighted sum
        else:
            output = F.adaptive_avg_pool2d(f, (1,1)).view(N,C) # global average pooling
        return a, output

class VGG16ImageEncoder(nn.Module):
    def __init__(self, weights, out_size):
        super(VGG16ImageEncoder, self).__init__()
        self.out_size = out_size
        self.model = torchvision.models.vgg16(weights=weights)

        # modify the last predict layer to output the desired dimension
        self.model.classifier = nn.Sequential(
            
            nn.BatchNorm1d(25088),
            nn.Linear(25088, 10000),
            nn.Tanh(),
            nn.Linear(10000, out_size),
            nn.Tanh()
        )


    def freeze_param(self):
        for param in self.model.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.model.features(x).reshape((x.shape[0], -1))
        out = self.model.classifier(out)
        return out

class AttnVGG(nn.Module):
    def __init__(self, num_classes, normalize_attn=False, dropout=None):
        super(AttnVGG, self).__init__()
        net = torchvision.models.vgg16_bn(pretrained=True)
        self.conv_block1 = nn.Sequential(*list(net.features.children())[0:6])
        self.conv_block2 = nn.Sequential(*list(net.features.children())[7:13])
        self.conv_block3 = nn.Sequential(*list(net.features.children())[14:23])
        self.conv_block4 = nn.Sequential(*list(net.features.children())[24:33])
        self.conv_block5 = nn.Sequential(*list(net.features.children())[34:43])
        self.pool = nn.AvgPool2d(7, stride=1)
        self.dpt = None
        if dropout is not None:
            self.dpt = nn.Dropout(dropout)
        self.cls = nn.Linear(in_features=512+512+256, out_features=num_classes, bias=True)
        
       # initialize the attention blocks defined above
        self.attn1 = AttentionBlock(256, 512, 256, 4, normalize_attn=normalize_attn)
        self.attn2 = AttentionBlock(512, 512, 256, 2, normalize_attn=normalize_attn)
        
       
        self.reset_parameters(self.cls)
        self.reset_parameters(self.attn1)
        self.reset_parameters(self.attn2)
    def reset_parameters(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0., 0.01)
                nn.init.constant_(m.bias, 0.)
    def forward(self, x):
        block1 = self.conv_block1(x)       # /1
        pool1 = F.max_pool2d(block1, 2, 2) # /2
        block2 = self.conv_block2(pool1)   # /2
        pool2 = F.max_pool2d(block2, 2, 2) # /4
        block3 = self.conv_block3(pool2)   # /4
        pool3 = F.max_pool2d(block3, 2, 2) # /8
        block4 = self.conv_block4(pool3)   # /8
        pool4 = F.max_pool2d(block4, 2, 2) # /16
        block5 = self.conv_block5(pool4)   # /16
        pool5 = F.max_pool2d(block5, 2, 2) # /32
        N, __, __, __ = pool5.size()
        
        g = self.pool(pool5).view(N,512)
        a1, g1 = self.attn1(pool3, pool5)
        a2, g2 = self.attn2(pool4, pool5)
        g_hat = torch.cat((g,g1,g2), dim=1) # batch_size x C
        if self.dpt is not None:
            g_hat = self.dpt(g_hat)
        out = self.cls(g_hat)

        return [out, a1, a2]
    
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx) -> None:
        super(TextEncoder, self).__init__()
        self.embed1 = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

    def forward(self, x):
        return self.embed1(x)


# class Attention(nn.Module):
#     def __init__(self, encoder_dim, decoder_dim, attention_dim):
#         super(Attention, self).__init__()
#         self.attention_dim = attention_dim
#         self.W = nn.Linear(decoder_dim,attention_dim)
#         self.U = nn.Linear(encoder_dim,attention_dim)
#         self.A = nn.Linear(attention_dim,1)

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
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, self.vocab_size, bias=True),
            nn.Sigmoid()
        )

    def forward(self, features, captions):
        b_size = captions.shape[0]
        token_softmaxs = []
        h, c = self.init_hidden(features)
        embeddings = self.text_encoder(captions)
        for w in range(captions.shape[1]):
            h, c = self.lstm_cell(embeddings[:, w, :], (h, c))
            out = self.fc(h)
            token_softmaxs.append(out)

        return torch.stack(token_softmaxs, dim=0).to(get_device())

    def init_hidden(self, features):
        return self.init_h(features), self.init_c(features)

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
    print(vgg)

    net50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    print(net50)
    # model = BaselineRNN(400, 8496, 3000, torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES, 3, 1)
    # print(model)
