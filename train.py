import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision
import torch
import os
import matplotlib.pyplot as plt
from dataset import FDataset, collate
from models import BaselineRNN
from utils import get_device


def train(epoch=10, batch_size=64, lr=0.002):
    device = get_device()
    BASE_DIR = f"{os.getcwd()}/data/flickr8k"

    

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device))
    ])

    # caption_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.to(device))
    # ])

    dataset = FDataset(
        root_dir=BASE_DIR + "/Images",
        capFilename=BASE_DIR + "/captions.txt",
        transform=img_transform
    )

    train_test_ratio = [0.7, 0.3]
    generator = torch.Generator().manual_seed(0)  # fix the seed for random_split
    train_set, test_set = random_split(dataset, train_test_ratio, generator=generator)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate
    )

    # test_loader = DataLoader(
    #     dataset=test_set,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     collate_fn=collate
    # )

    print("vocab_size:", dataset.vocab_size)
    model = BaselineRNN(400, dataset.vocab_size, torchvision.models.VGG16_Weights.DEFAULT, 3).to(device)
    model.train()
    model.img_encoder.freeze_param()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criteria = torch.nn.CrossEntropyLoss(ignore_index=0)

    loss_history = []
    fig = plt.figure()  # figsize=(5, 3)
    ax = fig.add_subplot(111)
    ax.set_xlabel("# batch")
    ax.set_ylabel("loss")
    plt.ion()

    fig.show()
    fig.canvas.draw()
    for e in range(epoch):
        for b, (imgs, captions) in enumerate(train_loader):
            optimizer.zero_grad()

            #b_size = len(captions)
            model.decoder.set_target_captions(captions)
            captions_softmaxs = model(imgs)
            output = captions_softmaxs.reshape((-1, captions_softmaxs.shape[-1]))
            target = captions.reshape(-1)

            print(output.shape)
            loss = criteria(output, target)
            loss_history.append(loss.item())
            loss.backward()
            optimizer.step()
            print(f"epoch {e}: {b},\tloss={round(loss.item(), 3)}")

            ax.clear()
            ax.plot(loss_history)
            fig.canvas.draw()
            print("loading next batch...")


if __name__ == "__main__":
    train()