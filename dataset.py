import os
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import tensorflow as tf
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from utils import get_device


class FDataset(Dataset):
    
    def __init__(self, root_dir, capFilename, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(capFilename)
        self.transform = transform
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        self.clean_data()
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<PAD>")
        self.tokenizer.fit_on_texts(self.captions)
        self.vocab_size = len(self.tokenizer.word_index)+1

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.tokenizer.texts_to_sequences([caption])[0]

    def clean_data(self):
        for i in range(len(self.captions)):
            caption = self.captions[i]
            caption = caption.split()
            caption = [word.lower() for word in caption]
            self.captions[i] = "startseq " + ' '.join(caption) + " endseq"


def collate(batch):
    imgs = []
    caps = []
    for item in batch:
        imgs.append(item[0].unsqueeze(0))
        caps.append(item[1])
    imgs = torch.cat(imgs,dim=0)
    caps = tf.keras.preprocessing.sequence.pad_sequences(caps, padding='post', value=1)
    #caps = pad_sequence(caps,batch_first=False,padding_value=1)
    return imgs, torch.from_numpy(caps).to(device=get_device(), dtype=torch.long)


        

'''if __name__ == "__main__":
    BASE_DIR = f"{os.getcwd()}/data/flickr8k"
    transformer = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor()])
    dataset =  FDataset(
        root_dir = BASE_DIR+"/Images",
        capFilename = BASE_DIR+"/captions.txt",
        transform=transformer
    )
    print(dataset.tokenizer.sequences_to_texts([[0,1,2,3,4]]))
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=True,
        collate_fn=collate
    )
    for i in dataloader:
        print(i)
        print(len(i[0]),len(i[1]))
        break
'''

