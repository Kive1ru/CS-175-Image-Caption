import torch


def get_device():
    # return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def save_model(epoch, batch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def load_model(path, model, optimizer):
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        batch = checkpoint['batch']
        loss = checkpoint['loss']
    except FileNotFoundError:
        epoch = 0
        batch = 0
        loss = 0

    return epoch, batch, loss
