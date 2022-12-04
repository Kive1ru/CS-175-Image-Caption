import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
import torchvision
import torch
import os
import matplotlib.pyplot as plt
from dataset import FDataset, TestDataset, collate, collate_1
from models import BaselineRNN, Img2Cap
from utils import get_device, save_model, load_model
from eval import generate_captions
import time
from similarity_check_tools import similarity_check_tool

BASE_DIR = f"{os.getcwd()}/data/flickr8k"
MODEL_PATH = "model_weights.torch"
NUM_WORKERS = 4


def plot_and_save(ax, fig, train_losses, eval_losses, eval_x_axis):
    ax.clear()
    ax.plot(train_losses, 'b', label="train")
    ax.plot(eval_x_axis, eval_losses, 'r', label="eval")
    ax.set_title("Image Captioning Transformer")
    ax.set_xlabel("# batch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.canvas.draw()
    fig.savefig("figs/loss.png")


def train(epochs=25, batch_size=128, lr=0.0003, num_layers=3):
    device = get_device()

    img_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_set = FDataset(
        root_dir=BASE_DIR + "/Images",
        capFilename=BASE_DIR + "/train.csv",
        transform=img_transform
    )

    test_set = TestDataset(
        root_dir=BASE_DIR + "/Images",
        capFilename=BASE_DIR + "/test.csv",
        transform=img_transform,
        tokenizer=train_set.tokenizer
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_1
    )

    # for img, caps in test_loader:
    #     print(img.shape)
    #     print(caps.shape)
    #     assert False

    print("vocab_size:", train_set.vocab_size)
    print(f"training on {device} with lr={lr}")
    # model = BaselineRNN(300, 512, num_layers, dataset.tokenizer, 2048,
    #                     torchvision.models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    model = Img2Cap(train_set.tokenizer, 400, torchvision.models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    model.train()
    model.img_encoder.freeze_param()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criteria = torch.nn.CrossEntropyLoss(ignore_index=train_set.tokenizer.word_index['<PAD>'])
    
    similarity_tool = similarity_check_tool()

    train_losses = []
    eval_losses = []
    bleu_scores = []
    eval_x_axis = []
    fig = plt.figure()  # figsize=(5, 3)
    ax = fig.add_subplot(111)
    plt.ion()

    fig.show()
    fig.canvas.draw()
    last_time = time.time()
    try:
        for e in range(epochs):
            # train
            model.train()
            for b, (imgs, captions) in enumerate(train_loader):
                imgs = imgs.to(device)
                captions = captions.to(device)
                optimizer.zero_grad()
                captions_softmaxs = model(imgs, captions[:, :-1])  # [batch_size, cap_len, vocab_size]
                loss = 0

                # ignore startseq for loss computation
                for w in range(captions.shape[1] - 1):
                    output = captions_softmaxs[:, w, :]
                    target = captions[:, w+1]
                    loss += criteria(output, target)
                loss = loss / (w + 1)
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()

                now = time.time()
                print(f"\repoch {e}: {b} loss={loss.detach().item()}. Took {round(now - last_time, 3)} seconds.")  # loss={round(loss.item(), 4)}
                last_time = now

                # validate
                if (b+1) % 100 == 0:
                    model.eval()
                    generate_captions(model, test_loader, train_set.tokenizer, similarity_tool, f"fig_{e}_{b}", img_num=5)
                    model.train()

                plot_and_save(ax, fig, train_losses, eval_losses, eval_x_axis)
            scheduler.step()

            # evaluate
            model.eval()
            eval_loss = []
            for b, (imgs, captions) in enumerate(test_loader):
                imgs = imgs.to(device)
                captions = captions.to(device)

                for i in range(5):
                    captions_softmaxs = model(imgs, captions[i::5, :-1])  # [batch_size, cap_len, vocab_size]

                    # ignore startseq for loss computation
                    for w in range(captions.shape[1] - 1):
                        output = captions_softmaxs[:, w, :]
                        target = captions[i::5, w + 1]
                        l = criteria(output, target)

                        if l.item() == l.item():  # filter out NaN  # TODO: why is it NaN only while trained on CUDA?
                            eval_loss.append(l.item())

            eval_losses.append(sum(eval_loss) / len(eval_loss))  # only plot a loss for each epoch
            eval_x_axis.append(len(train_losses))
            bleu, similarity = generate_captions(model, test_loader, train_set.tokenizer, similarity_tool, f"fig_{e}")
            bleu_scores.append(bleu)
            print("eval loss:", sum(eval_loss) / len(eval_loss), "bleu score:", bleu, "similarity:", similarity)

            plot_and_save(ax, fig, train_losses, eval_losses, eval_x_axis)

            save_model(epochs, model, optimizer, loss, f"model_weights/caption_{e}.torch")
            torch.save({
                "eval_losses": eval_losses,
                "bleu_scores": bleu_scores,
                "eval_x_axis": eval_x_axis,
                "train_losses": train_losses
            }, "loss_info.torch")

    except:
        # save_model(e, b, model, optimizer, loss, MODEL_PATH)
        raise



if __name__ == "__main__":
    train()
