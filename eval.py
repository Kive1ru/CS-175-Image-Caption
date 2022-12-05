import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from utils import get_device
from models import Img2Cap
from dataset import FDataset
from nltk.translate.bleu_score import sentence_bleu
import re
from pathlib import Path

BASE_DIR = "data/flickr8k"


def evaluate_model(model, description, pictures, tokenizer, max_cap):
    actuallist, predictlist = list(), list()
    for key, desc_list in description.items():
        prediction = ...
        actual_des = ...
        actuallist.append(actual_des)
        predictlist.append(prediction)

    print("")

def generate_captions(model, dataloader, tokenizer, similarity_checker, file_prefix, img_num=None):
    """
    Returns the average BLEU score and similarity score over all the image and caption targets in the dataloader.
    """
    device = get_device()
    model = model.to(device)
    count = 0
    bleu_scores = []
    similarities = []
    for imgs, captions in dataloader:
        b_size = imgs.shape[0]
        imgs = imgs.to(device)
        for i in range(b_size):
            if img_num is not None and count >= img_num:
                return
            seq = model.predict(imgs[i].reshape((1, imgs.shape[1], imgs.shape[2], imgs.shape[3])))
            sentence = beautify(tokenizer.sequences_to_texts([seq])[0])
            target_sentences = [beautify(target_caption) for target_caption in tokenizer.sequences_to_texts(captions[i*5:(i+1)*5].cpu().detach().tolist())]

            ngram = min(4, min(len(s) for s in [sentence] + target_sentences))  # set the largest n-gram to be 4-gram
            bleu_score = sentence_bleu([s.split() for s in target_sentences], sentence.split(), weights=[1/ngram for _ in range(ngram)])
            bleu_scores.append(bleu_score)

            if similarity_checker is not None:
                _, similarity = similarity_checker.check_sentences([sentence] + target_sentences)
                similarities.append(similarity)

            if img_num is not None:
                print(target_sentences)
                print(sentence)
                print(f"BLEU score diff: {bleu_score}")
                print()
                save_image_caption(
                    imgs[i].cpu().detach().numpy(),
                    sentence + f" BLEU score: {bleu_score}",
                    f"figs/{file_prefix}{count}.png"
                )
            count += 1
    return sum(bleu_scores) / len(bleu_scores), sum(similarities) / len(similarities)


def save_image_caption(img, caption, file_path=None):
    """
    Parameters
    ----------
    img: np.ndarray
        (3xHxW)
    caption: string
    """
    count = sum(1 for _ in Path("figs").glob("fig_*_*.png"))
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406

    img = img.transpose((1, 2, 0))

    plt.ioff()
    fig = plt.figure()  # figsize=(5, 3)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    # ax.text(10, 220, caption)
    ax.set_title(caption)
    if file_path is not None:
        fig.savefig(file_path)
    plt.ion()


def beautify(sentence: str) -> str:
    ans = re.sub(r' <EOS>( <PAD>)*', '.', sentence)
    ans = re.sub(r'<SOS> ', '', ans)
    ans = ans[0].upper() + ans[1:]
    return ans


def experiment():
    img_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = FDataset(
        root_dir=BASE_DIR + "/Images",
        capFilename=BASE_DIR + "/captions.txt",
        transform=img_transform,
        num_words=None
    )

    tokenizer = dataset.tokenizer
    device = get_device()
    weight_path = "model_weights/caption_24.torch"
    model = Img2Cap(tokenizer, 400, torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu'))["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_images_dir = Path("/Users/jackyu/Desktop/images")

    for img_path in test_images_dir.iterdir():
        print(img_path, ":")
        try:
            img = Image.open(img_path).convert("RGB")
            # img = plt.imread(img_path)[:, :, :3]
            # print(img.shape)
            img = img_transform(img)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2])).to(device)

            seq = model.predict(img)
            sentence = beautify(tokenizer.sequences_to_texts([seq])[0])  # list of strings
            print(sentence)
            print()
            save_image_caption(img[0].cpu().detach().numpy(), sentence, "figures.png")
        except Exception as err:
            print(err)
            # raise


if __name__ == "__main__":
    # experiment()
    pass
    l1 = [1,2,3,2,1,2,3,2,1]
    l2 = [5,7,7,6,5,9,7,6,6]
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(l1, 'b')
    ax.set_ylabel("l1")
    ax2.plot(l2, 'r')
    ax2.set_ylabel("l2")
    plt.show()

    # ngram = 3
    # score = sentence_bleu(["this is".split()], "this is me shouting".split(), [1/ngram for _ in range(ngram)])
    # print(score)
