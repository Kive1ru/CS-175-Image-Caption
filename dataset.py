import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from utils import get_device
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.model_selection import train_test_split


class FDataset(Dataset):

    def __init__(self, root_dir, capFilename, transform=None, num_words=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(capFilename)
        self.transform = transform
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        self.clean_data()
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<UNK>", num_words=num_words, lower=False, filters='')
        self.tokenizer.fit_on_texts(self.captions)
        self.tokenizer.word_index['<PAD>'] = 0
        self.tokenizer.index_word[0] = '<PAD>'
        self.vocab_size = len(self.tokenizer.word_index) if num_words is None else num_words + 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(self.tokenizer.texts_to_sequences([caption])[0])

    def clean_data(self):
        for i in range(len(self.captions)):
            caption = self.captions[i]
            caption = caption.split()
            caption = [word.lower() for word in caption if len(word) > 1 and word.isalpha()]
            self.captions[i] = "<SOS> " + ' '.join(caption) + " <EOS>"


class TestDataset(Dataset):
    def __init__(self, root_dir, capFilename, transform=None, tokenizer=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(capFilename)
        self.transform = transform
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        self.randImg = list(set(self.imgs))
        self.capDict = {}
        self.clean_data()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.randImg)
    
    def __getitem__(self,idx):
        img_name = self.randImg[idx]
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location)
        if self.transform is not None:
            img = self.transform(img)
        captionlist = self.capDict[img_name]
        captionlist = [torch.tensor(self.tokenizer.texts_to_sequences([cap])[0]) for cap in captionlist]
        return img, captionlist
    
    def clean_data(self):
        for i in range(len(self.captions)):
            caption = self.captions[i]
            caption = caption.split()
            caption = [word.lower() for word in caption if len(word)>1 and word.isalpha()]
            self.captions[i] = "<SOS> " + ' '.join(caption) + " <EOS>"
            if self.imgs[i] not in self.capDict:
                self.capDict[self.imgs[i]] = [self.captions[i]]
            else:
                self.capDict[self.imgs[i]].append(self.captions[i])


def splitDataset():
    BASE_DIR = f"{os.getcwd()}/data/flickr8k"
    capFilename = BASE_DIR + "/captions.txt"
    df = pd.read_csv(capFilename)
    #imgs = np.array(df["image"])
    #train, test = train_test_split(imgs, test_size=0.75)
    train, test = train_test_split(df, test_size=0.3, shuffle=False)
    train.to_csv(BASE_DIR + "/train.csv", index=False)
    test.to_csv(BASE_DIR + "/test.csv", index=False)


def collate_train(batch):
    imgs = []
    caps = []
    for item in batch:
        imgs.append(item[0].unsqueeze(0))
        caps.append(item[1])
    imgs = torch.cat(imgs, dim=0)
    caps = pad_sequence(caps, batch_first=True, padding_value=0)
    return imgs, caps

def collate_test(batch):
    imgs = []
    caps = []
    for item in batch:
        imgs.append(item[0].unsqueeze(0))
        for i in range(len(item[1])):
            caps.append(item[1][i])
    imgs = torch.cat(imgs, dim=0)
    caps = pad_sequence(caps, batch_first=True, padding_value=0)
    return imgs, caps


if __name__ == "__main__":
    '''
    BASE_DIR = f"{os.getcwd()}/data/flickr8k"
    transformer = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor()])
    dataset = FDataset(
        root_dir=BASE_DIR + "/Images",
        capFilename=BASE_DIR + "/captions.txt",
        transform=transformer
    )
    # print(dataset[0][1].tolist())
    # print(dataset.tokenizer.sequences_to_texts([dataset[0][1].tolist()]))
    print(dataset.tokenizer.index_word)
    print(dataset.tokenizer.index_word[0])
    print(dataset.tokenizer.index_word[1])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=True,
        collate_fn=collate
    )
    for i in dataloader:
        print(i[1])
        #print(len(i[0]),len(i[1]))
        break
        '''
    # splitDataset()
        