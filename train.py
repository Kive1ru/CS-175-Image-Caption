import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch
import os
from dataset import FDataset, collate
from models import BaselineRNN
from utils import get_device

def train():
    device = get_device()
    BASE_DIR = f"{os.getcwd()}/data/flickr8k"

    epoch = 10
    batch_size = 20
    lr = 0.005

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

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate
    )

    print("vocab_size:", dataset.vocab_size)
    model = BaselineRNN(400, dataset.vocab_size, torchvision.models.VGG16_Weights.DEFAULT, 3).to(device)
    model.train()
    model.img_encoder.freeze_param()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criteria = torch.nn.CrossEntropyLoss(ignore_index=0)

    loss_history = []
    for e in range(epoch):
        for b, (imgs, captions) in enumerate(dataloader):
            optimizer.zero_grad()

            #b_size = len(captions)
            model.decoder.set_target_captions(captions)
            captions_softmaxs = model(imgs)
            output = captions_softmaxs.reshape((-1, captions_softmaxs.shape[-1]))
            target = captions.reshape(-1)

            print(output.shape)
            loss = criteria(output, target)
            loss_history.append(loss)
            loss.backward()
            optimizer.step()

            print('\r', f"epoch {e}: {b},\tloss={round(loss.item(), 3)}", end=' ')


if __name__ == "__main__":
    train()