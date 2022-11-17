import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from PIL import Image
from pathlib import Path
from utils import get_device
from nltk.translate.bleu_score import sentence_bleu


def evaluate_model(model, description, pictures, tokenizer, max_cap):
    actuallist, predictlist = list(), list()
    for key, desc_list in description.items():
        prediction = ...
        actual_des = ...
        actuallist.append(actual_des)
        predictlist.append(prediction)
        
        
    print("")


def generate_captions(model, dataloader, tokenizer):
    device = get_device()
    model = model.to(device)
    for imgs, captions in dataloader:
        b_size = captions.shape[0]
        imgs = imgs.to(device)
        for i in range(b_size):
            seq = model.predict(imgs[i].reshape((1, imgs.shape[1], imgs.shape[2], imgs.shape[3])))
            sentence = tokenizer.sequences_to_texts([seq])[0] # list of strings
            cap_sentence = tokenizer.sequences_to_texts(captions[i].reshape((1, captions.shape[1])).cpu().detach().tolist())[0]
            bleu_score = sentence_bleu(cap_sentence.split(), sentence.split(), weights=(0.5, 0.5))
            print(cap_sentence)
            print(sentence)
            print(f"BLEU score diff: {bleu_score}")
            print()



# def generate_captions(model, dataloader, tokenizer):
#     device = get_device()
#     model = model.to(device)
#     imgs, captions = next(dataloader)
#     imgs = imgs.to(device)
#     seqs = model.predict(imgs)
#     sentences = tokenizer.sequences_to_texts(seqs.cpu().detach().tolist()) # list of strings
#     cap_sentences = tokenizer.sequences_to_texts(captions.tolist())
#
#     imgs = imgs.cpu()
#     for s in range(len(sentences)):
#         bleu_score = sentence_bleu(cap_sentences[s], sentences[s], weights=(0.5, 0.5))
#         print(cap_sentences[s])
#         print(sentences[s])
#         print(f"BLEU score diff: {bleu_score}")
#         print()

def show_image_caption(img, caption):
    """
    Parameters
    ----------
    img: np.ndarray
        (3xHxW)
    caption: string
    """
    img
    plt.imshow(img)