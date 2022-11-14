import os
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import tensorflow as tf

class FDataset(Dataset):
    
    def __init__(self, root_dir, capFilename, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(capFilename)
        self.transform = transform
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(self.captions)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        caption = self.captions[idx]
        img_name = "Images/" + self.imgs[idx]
        img_location = os.path.join(self.root_dir,img_name)
        img = tf.io.read_file(img_location)
        if self.transform is not None:
            img = self.transform(img)
        caption = caption.split()
        caption = [word.lower() for word in caption]
        caption = "startseq " + ' '.join(caption) + " endseq"
        return img, self.tokenizer.texts_to_sequences([caption])[0]
    
