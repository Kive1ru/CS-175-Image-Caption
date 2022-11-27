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


def generate_captions(model, dataloader, tokenizer, epoch, batch, img_num=5):
    device = get_device()
    model = model.to(device)
    count = 0
    for imgs, captions in dataloader:
        b_size = captions.shape[0]
        imgs = imgs.to(device)
        for i in range(b_size):
            if count >= img_num:
                return
            seq = model.predict(imgs[i].reshape((1, imgs.shape[1], imgs.shape[2], imgs.shape[3])))
            sentence = tokenizer.sequences_to_texts([seq])[0]  # list of strings
            target_sentence = tokenizer.sequences_to_texts(captions[i].reshape((1, captions.shape[1])).cpu().detach().tolist())[0]
            bleu_score = sentence_bleu(target_sentence.split(), sentence.split(), weights=(0.5, 0.5))
            print(target_sentence)
            print(sentence)
            print(f"BLEU score diff: {bleu_score}")
            print()
            count += 1
            save_image_caption(
                imgs[i].cpu().detach().numpy(),
                "\n".join([target_sentence, sentence, f"BLEU score diff: {bleu_score}"]),
                f"figs/fig_{epoch}_{batch}_{count}.png"
            )



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

def save_image_caption(img, caption, file_path):
    """
    Parameters
    ----------
    img: np.ndarray
        (3xHxW)
    caption: string
    """
    count = sum(1 for _ in Path("figs").glob("fig_*_*.pnj"))
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406

    img = img.transpose((1, 2, 0))
    # R, G, B = img[0], img[1], img[2]
    # img = np.stack((R, G, B), axis=2)
    print("img.max():", img.max())

    plt.ioff()
    fig = plt.figure()  # figsize=(5, 3)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.set_title(caption)
    fig.savefig(file_path)
    plt.ion()
