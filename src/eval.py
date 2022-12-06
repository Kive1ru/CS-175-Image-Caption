import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from utils import get_device, load_model
from models import Img2Cap
from dataset import FDataset
from nltk.translate.bleu_score import sentence_bleu
import re
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def evaluate_models(models_folder, num_models_start, num_models_end, file_prefix, extension, dataloader, tokenizer, similarity_checker):
    device = get_device()
    model_dir = Path(models_folder)
    assert model_dir.is_dir()

    bleu_scores = []
    similarities = []
    model = Img2Cap(tokenizer, 400, torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    for e in range(num_models_start, num_models_end):
        model = load_model(model_dir / (file_prefix + str(e) + extension), model)
        # model.load_state_dict(torch.load(model_dir / (file_prefix + str(e) + extension))["model_state_dict"])
        bleu_score, similarity = generate_captions(model, dataloader, tokenizer, similarity_checker, f"epoch{e}_")
        bleu_scores.append(bleu_score)
        similarities.append(similarity)
        print(f"epoch {e}: BLEU score={round(bleu_score, 3)}, similarity={round(similarity, 3)}")
    return bleu_scores, similarities

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
                    sentence,
                    f"figs/{file_prefix}{count}.png"
                )
            count += 1
            if count % 200 == 0:
                print(count, sum(bleu_scores) / len(bleu_scores))
    return sum(bleu_scores) / len(bleu_scores), sum(similarities) / len(similarities) if len(similarities) > 0 else 0


def save_image_caption(img, caption, file_path=None):
    """
    Parameters
    ----------
    img: np.ndarray
        (3xHxW)
    caption: str
    """
    count = sum(1 for _ in Path("../figs").glob("fig_*_*.png"))
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
    ax.set_title(caption, wrap=True)
    if file_path is not None:
        fig.savefig(file_path)
    plt.ion()


def beautify(sentence: str) -> str:
    ans = re.sub(r' <EOS>( <PAD>)*', '.', sentence)
    ans = re.sub(r'<SOS> ', '', ans)
    ans = ans[0].upper() + ans[1:]
    return ans


def experiment(img_folder, dataset="8K"):
    if dataset == "8K":
        base_dir = "../data/flickr8k"
        num_words = None
        weights_path = "../model_weights/8K/caption_24.torch"
    elif dataset == "30K":
        base_dir = "../data/flickr30k"
        num_words = 11181
        weights_path = "../model_weights/30K/caption_24.torch"
    else:
        raise ValueError

    img_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_set = FDataset(
        root_dir=base_dir + "/Images",
        capFilename=base_dir + "/train.csv",  # captions.txt
        transform=img_transform,
        num_words=num_words
    )

    tokenizer = train_set.tokenizer
    device = get_device()
    model = Img2Cap(tokenizer, 400, torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_images_dir = Path(img_folder)

    for img_path in test_images_dir.iterdir():
        print(img_path, ":")
        if not img_path.is_file():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            actual_img = ImageOps.exif_transpose(img)  # pass this to visualization, but not model's input because model is trained on images without rotation correction
            # img = plt.imread(img_path)[:, :, :3]
            # print(img.shape)
            img = img_transform(img)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2])).to(device)
            actual_img = img_transform(actual_img)
            actual_img = actual_img.reshape((1, actual_img.shape[0], actual_img.shape[1], actual_img.shape[2])).to(device)

            seq = model.predict(img)
            sentence = beautify(tokenizer.sequences_to_texts([seq])[0])
            print(sentence)
            print()
            save_image_caption(actual_img[0].cpu().detach().numpy(), sentence, f"../figs/figures_{dataset}_{img_path.stem}.png")
        except Exception as err:
            print(err)
            # raise


if __name__ == "__main__":
    experiment("/Users/jackyu/Desktop/images", "8K")
    pass
