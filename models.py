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
        avgpool = self.model.avgpool(features)
        avgpool = avgpool.reshape((avgpool.shape[0], -1))  # avgpool.shape[0] is the batch size
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
    def __init__(self, embed_dim, vocab_size, lstm_layer_num, EOS_token_id, max_token_num=35):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.lstm_layer_num = lstm_layer_num
        self.EOS_token_id = EOS_token_id
        self.max_token_num = max_token_num
        self.is_training = False
        self.text_encoder = TextEncoder(self.vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.vocab_size, num_layers=self.lstm_layer_num)

    def forward(self, feature_vector):
        device = get_device()
        batch_size = feature_vector.shape[0]
        lstm_input = feature_vector.reshape((1, batch_size, self.embed_dim))
        token_softmax = []
        all_token_ids = None
        h_n = torch.zeros((self.lstm_layer_num, batch_size, self.vocab_size)).to(device)
        c_n = torch.zeros((self.lstm_layer_num, batch_size, self.vocab_size)).to(device)

        reached_EOS = False
        while not reached_EOS and len(token_softmax) < self.target_captions.shape[1] if self.is_training else self.max_token_num:
            out, (h_n, c_n) = self.lstm(lstm_input, (h_n, c_n))
            prev_token_id = out.argmax(dim=2)  # dim=1 is the batch dimension
            # print("prev_token_id.shape:", prev_token_id.shape)
            # print("all_token_ids.shape:", all_token_ids.shape)
            if all_token_ids is None:
                all_token_ids = prev_token_id.transpose(0, 1)
            else:
                all_token_ids = torch.cat((all_token_ids, prev_token_id.transpose(0, 1)), dim=1)
                
            if self.is_training:
                lstm_input = self.text_encoder(self.target_captions[:, len(token_softmax)]).reshape((1, batch_size, self.embed_dim))
            else:
                lstm_input = self.text_encoder(prev_token_id.to(device=device, dtype=int)).reshape((1, batch_size, self.embed_dim))
            token_softmax.append(out)
            reached_EOS = (all_token_ids == self.EOS_token_id).any(dim=1).to(torch.float32).mean() == 1
        
        # append padding
        if self.is_training:
            self.is_training = False
            for _ in range(self.target_captions.shape[1] - len(token_softmax)):
                token_softmax.append(torch.zeros((1, batch_size, self.vocab_size)).to(device))
                
        return torch.stack(token_softmax, dim=0)
        


    def predict(self, feature_vector):
        token_softmaxs = self.forward(feature_vector)
        token_ids = token_softmaxs.argmax(dim=1)
        return token_ids

    def set_target_captions(self, captions):
        """
        Parameters
        ----------
        captions: torch.Tensor
            2D array with shape (B x L). B = batch size; L = sentence length
        """
        self.is_training = True
        self.target_captions = captions


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
    # img_cap_model = BaselineRNN(400, 2000, torchvision.models.VGG16_Weights.DEFAULT, 3).to(device)

