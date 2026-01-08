import torch
import torch.nn.functional as F
import numpy as np


def dice_loss(pred, target, smooth=1e-6):
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    inter = (pred * target_oh).sum((2, 3))
    union = pred.sum((2, 3)) + target_oh.sum((2, 3))
    return 1 - ((2 * inter + smooth) / (union + smooth)).mean()


def combined_loss_fn(outputs, targets, ce_fn, dice_w=1.0):
    return ce_fn(outputs, targets) + dice_w * dice_loss(outputs, targets)


def compute_metrics(preds, targets, smooth=1e-6):
    pixel_acc = (preds == targets).mean()
    inter = (preds & targets).sum()
    dice = (2 * inter + smooth) / (preds.sum() + targets.sum() + smooth)
    union = preds.sum() + targets.sum() - inter
    iou = (inter + smooth) / (union + smooth)
    return pixel_acc, dice, iou


def train_segmentation(model, loader, loss_fn, optimizer, device):
    model.train()
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_segmentation(model, loader, loss_fn, device):
    model.eval()
    total, P, T = 0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        total += loss.item() * x.size(0)
        P.append(out.argmax(1).cpu())
        T.append(y.cpu())
    return total / len(loader.dataset), torch.cat(P).numpy(), torch.cat(T).numpy()
