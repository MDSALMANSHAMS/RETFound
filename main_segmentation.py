import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download
from util.pos_embed import interpolate_pos_embed

from models_segmentation import RETFoundSegmentation
from engine_segmentation import (
    train_segmentation,
    evaluate_segmentation,
    combined_loss_fn,
    compute_metrics,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# Dataset
# =========================
class OCTDrusenDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.images = sorted(os.listdir(self.image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_dir, self.images[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, self.images[idx]), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        return img, mask.long()


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser("RETFound Segmentation")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--drop_path", type=float, default=0.2)
    parser.add_argument("--finetune", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./segmentation_output")
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--ce_weight", type=str, default="0.3,0.7")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    transform = Compose([
        Resize(args.img_size, args.img_size),
        Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ToTensorV2()
    ])

    train_ds = OCTDrusenDataset(os.path.join(args.data_path, "train"), transform)
    val_ds   = OCTDrusenDataset(os.path.join(args.data_path, "val"), transform)
    test_ds  = OCTDrusenDataset(os.path.join(args.data_path, "test"), transform)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=4)

    model = RETFoundSegmentation(args.img_size, args.patch_size).to(device)

    if args.finetune:
        ckpt = hf_hub_download(f"YukunZhou/{args.finetune}", f"{args.finetune}.pth")
        state = torch.load(ckpt, map_location="cpu")
        state = state["model"] if "model" in state else state
        interpolate_pos_embed(model.encoder, state)
        model.encoder.load_state_dict(state, strict=False)

    ce_weights = torch.tensor([float(x) for x in args.ce_weight.split(",")]).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=ce_weights)

    def loss_fn(out, tgt):
        return combined_loss_fn(out, tgt, ce_loss, args.dice_weight)

    opt = optim.AdamW(model.parameters(), lr=args.lr)

    best = 1e9
    for e in range(args.epochs):
        train_loss = train_segmentation(model, train_loader, loss_fn, opt, device)
        val_loss, P, T = evaluate_segmentation(model, val_loader, loss_fn, device)
        acc, dice, iou = compute_metrics(P, T)

        print(f"Epoch {e+1}: Train={train_loss:.4f} Val={val_loss:.4f} Dice={dice:.4f} IoU={iou:.4f}")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), f"{args.output_dir}/best.pth")

    test_loss, P, T = evaluate_segmentation(model, test_loader, loss_fn, device)
    acc, dice, iou = compute_metrics(P, T)
    print(f"Test: Loss={test_loss:.4f} Dice={dice:.4f} IoU={iou:.4f}")


if __name__ == "__main__":
    main()
