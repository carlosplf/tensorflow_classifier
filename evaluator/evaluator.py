import torch

@torch.no_grad()
def evaluate(model, device, val_loader):
    print("Len validation loader: ", len(val_loader))
    model.eval()
    outputs = [model.validation_step(batch, device) for batch in val_loader]
    return model.validation_epoch_end(outputs)