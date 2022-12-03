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
            print("seq:", seq)
            sentence = beautify(tokenizer.sequences_to_texts([seq])[0])
            target_sentence = beautify(
                tokenizer.sequences_to_texts(captions[i].reshape((1, captions.shape[1])).cpu().detach().tolist())[0])
            bleu_score = sentence_bleu([target_sentence.split()], sentence.split(), weights=(0.5, 0.5))
            print(target_sentence)
            print(sentence)
            print(f"BLEU score diff: {bleu_score}")
            print()
            count += 1
            save_image_caption(
                imgs[i].cpu().detach().numpy(),
                sentence + f" BLEU score: {bleu_score}",
                f"figs/fig_{epoch}_{batch}_{count}.png"
            )


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
        transform=img_transform
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
            raise


if __name__ == "__main__":
    pass
    # ngram = 3
    # score = sentence_bleu(["this is".split()], "this is me shouting".split(), [1/ngram for _ in range(ngram)])
    # print(score)
