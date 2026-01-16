import os
import cv2
import torch
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from models_segmentation import RETFoundSegmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Metrics
# ============================================================
def dice_iou(pred, gt, smooth=1e-6):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    inter = (pred & gt).sum()
    union = (pred | gt).sum()

    dice = (2 * inter + smooth) / (pred.sum() + gt.sum() + smooth)
    iou  = (inter + smooth) / (union + smooth)

    return dice, iou


# ============================================================
# Model loader
# ============================================================
def load_model(ckpt, img_size=256):
    model = RETFoundSegmentation(img_size=img_size).to(DEVICE)

    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)

    model.eval()
    return model


# ============================================================
# Preprocess
# ============================================================
def preprocess(img_path, img_size=256):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tf = Compose([
        Resize(img_size, img_size),
        Normalize((0.485, 0.456, 0.406),
                  (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    aug = tf(image=img)
    return aug["image"].unsqueeze(0), img


# ============================================================
# Overlay visualization
# ============================================================
def make_overlay(img, mask):
    # Ensure correct type
    mask = mask.astype(np.uint8)

    mask = cv2.resize(
        mask,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    color = np.zeros_like(img, dtype=np.uint8)
    color[:, :, 1] = mask * 255      # Green channel

    overlay = cv2.addWeighted(img, 0.7, color, 0.3, 0)
    return overlay


# ============================================================
# Main inference
# ============================================================
def run_inference(
    ckpt="segmentation_output/best.pth",
    test_img_dir="Segmentation/Data/test/images",
    test_mask_dir="Segmentation/Data/test/masks",
    out_dir="segmentation_output/inference",
    img_size=256
):

    os.makedirs(out_dir, exist_ok=True)

    model = load_model(ckpt, img_size)

    dice_scores = []
    iou_scores  = []

    print("\n===== Running Inference =====\n")

    for name in sorted(os.listdir(test_img_dir)):

        img_path = os.path.join(test_img_dir, name)

        stem = os.path.splitext(name)[0]
        gt_name = stem + "_mask.png"
        gt_path = os.path.join(test_mask_dir, gt_name)

        if not os.path.isfile(gt_path):
            continue

        # ----- Predict -----
        tensor, orig = preprocess(img_path, img_size)

        with torch.no_grad():
            out = model(tensor.to(DEVICE))
            pred = out.argmax(1).squeeze().cpu().numpy()

        # ----- Load GT -----
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = (gt > 0).astype("uint8")

        # Align GT to prediction resolution
        gt = cv2.resize(
            gt,
            (pred.shape[1], pred.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # ----- Metrics -----
        dice, iou = dice_iou(pred, gt)
        dice_scores.append(dice)
        iou_scores.append(iou)

        # ----- Save outputs -----
        over = make_overlay(orig, pred)
        cv2.imwrite(os.path.join(out_dir, stem + "_overlay.png"), over)

        print(f"{name:25s}  Dice: {dice:.4f}   IoU: {iou:.4f}")

    # ---------------------------------------------------------
    print("\n===== FINAL REPORT =====")
    print("Mean Dice:", round(np.mean(dice_scores), 4))
    print("Mean IoU :", round(np.mean(iou_scores), 4))


if __name__ == "__main__":
    run_inference()
