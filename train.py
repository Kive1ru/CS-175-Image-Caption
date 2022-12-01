import torchvision.transforms as T
from torch.utils.data import DataLoader
import torchvision
import torch
import os
import matplotlib.pyplot as plt
from dataset import FDataset, collate
from models import BaselineRNN
from utils import get_device, save_model, load_model
from eval import generate_captions
import time


MODEL_PATH = "model_weights.torch"
NUM_WORKERS = 8


import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset

def random_split(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]



def train(epochs=10, batch_size=128, lr=0.0003, num_layers=3):
    device = get_device()
    BASE_DIR = f"{os.getcwd()}/data/flickr8k"

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
    train_test_ratio = [6000*5, 2091*5]
    generator = torch.Generator().manual_seed(0)  # fix the seed for random_split
    train_set, test_set = random_split(dataset, train_test_ratio, generator=generator)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate
    )

    print("vocab_size:", dataset.vocab_size)
    print(f"training on {device} with lr={lr}")
    model = BaselineRNN(300, 512, num_layers, dataset.tokenizer, 2048,
                        torchvision.models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    model.train()
    model.img_encoder.freeze_param()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criteria = torch.nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.word_index['<PAD>'])

    loss_history = []
    fig = plt.figure()  # figsize=(5, 3)
    ax = fig.add_subplot(111)
    ax.set_xlabel("# batch")
    ax.set_ylabel("loss")
    plt.ion()

    fig.show()
    fig.canvas.draw()
    last_time = time.time()
    try:
        for e in range(epochs):
            for b, (imgs, captions) in enumerate(train_loader):
                # print("done")
                imgs = imgs.to(device)
                captions = captions.to(device)
                optimizer.zero_grad()
                captions_softmaxs = model(imgs, captions[:, :-1])
                # print("computing loss...", end="")
                loss = 0

                # ignore startseq for loss computation
                for w in range(captions.shape[1] - 1):
                    output = captions_softmaxs[w, :, :]
                    target = captions[:, w+1]
                    loss += criteria(output, target)
                loss = loss / (w + 1)
                # print("done")
                loss_history.append(loss.item())
                # print("computing gradient...", end="")
                loss.backward()
                # print("done")
                # print("updating parameters...", end="")
                optimizer.step()
                # print("done")
                ax.clear()
                ax.plot(loss_history)
                fig.canvas.draw()

                now = time.time()
                print(f"\repoch {e}: {b} loss={loss.item()}. Took {round(now - last_time, 3)} seconds.")  # loss={round(loss.item(), 4)}
                last_time = now

                # validate
                if (b+1) % 100 == 0:
                    model.eval()
                    generate_captions(model, test_loader, dataset.tokenizer, e, b)

                # print("loading next batch...", end="")
                model.train()
            scheduler.step()
            save_model(epochs, model, optimizer, loss, f"model_weights/caption_{e}.torch")
        fig.savefig("figs/loss.png")
    except:
        # save_model(e, b, model, optimizer, loss, MODEL_PATH)
        raise

    # model.generate_captions(imgs)
    # texts = dataset.tokenizer.sequences_to_texts([[5, 2, 7, 9, 3], [10, 24, 543, 2, 6, 8]])
    # print(texts)


if __name__ == "__main__":
    train()
