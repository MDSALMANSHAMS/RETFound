import os
import cv2
import torch
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from models_segmentation import RETFoundSegmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Metric utilities
# ============================================================

def basic_iou(pred, gt, smooth=1e-6):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    inter = (pred & gt).sum()
    union = (pred | gt).sum()

    return (inter + smooth) / (union + smooth)


def dice_score(pred, gt, smooth=1e-6):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    inter = (pred & gt).sum()
    return (2 * inter + smooth) / (pred.sum() + gt.sum() + smooth)


def tolerance_iou(pred, gt, k=2):
    kernel = np.ones((k*2+1, k*2+1), np.uint8)
    gt_dilated = cv2.dilate(gt.astype(np.uint8), kernel)
    return basic_iou(pred, gt_dilated)


def lesion_level_iou(pred, gt):
    num_gt, gt_labels = cv2.connectedComponents(gt.astype(np.uint8))
    num_pr, pr_labels = cv2.connectedComponents(pred.astype(np.uint8))

    scores = []
    for g in range(1, num_gt):
        gt_comp = (gt_labels == g)

        best = 0
        for p in range(1, num_pr):
            pr_comp = (pr_labels == p)
            best = max(best, basic_iou(pr_comp, gt_comp))

        scores.append(best)

    if len(scores) == 0:
        return 1.0 if pred.sum() == 0 else 0.0

    return np.mean(scores)


def positive_slice_iou(pred, gt):
    if gt.sum() == 0:
        return None
    return basic_iou(pred, gt)


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
        Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    aug = tf(image=img)
    return aug["image"].unsqueeze(0), img


# ============================================================
# Overlay visualization
# ============================================================
def make_overlay(img, mask):
    mask = mask.astype(np.uint8)

    mask = cv2.resize(
        mask,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    color = np.zeros_like(img, dtype=np.uint8)
    color[:, :, 1] = mask * 255     # green channel

    return cv2.addWeighted(img, 0.7, color, 0.3, 0)


# ============================================================
# MAIN
# ============================================================
def run_inference(
    ckpt="segmentation_output/best.pth",
    test_img_dir="Segmentation/Data/test/images",
    test_mask_dir="Segmentation/Data/test/masks",
    out_dir="segmentation_output/inference_overlays",
    img_size=256
):

    os.makedirs(out_dir, exist_ok=True)

    model = load_model(ckpt, img_size)

    results = {
        "pixel_iou": [],
        "dice": [],
        "tol2_iou": [],
        "tol3_iou": [],
        "lesion_iou": [],
        "positive_iou": []
    }

    print("\n===== Multi-Metric Inference (Overlay Only) =====\n")

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

        gt = cv2.resize(
            gt,
            (pred.shape[1], pred.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # ----- Metrics -----
        piou  = basic_iou(pred, gt)
        dice  = dice_score(pred, gt)
        t2    = tolerance_iou(pred, gt, k=2)
        t3    = tolerance_iou(pred, gt, k=3)
        liou  = lesion_level_iou(pred, gt)
        pos   = positive_slice_iou(pred, gt)

        results["pixel_iou"].append(piou)
        results["dice"].append(dice)
        results["tol2_iou"].append(t2)
        results["tol3_iou"].append(t3)
        results["lesion_iou"].append(liou)

        if pos is not None:
            results["positive_iou"].append(pos)

        # ----- SAVE ONLY OVERLAY -----
        over = make_overlay(orig, pred)
        cv2.imwrite(os.path.join(out_dir, stem + "_overlay.png"), over)

        print(f"{name:22s}  IoU:{piou:.3f}  Dice:{dice:.3f}  "
              f"Tol2:{t2:.3f}  Lesion:{liou:.3f}")

    # ========================================================
    print("\n===== FINAL SUMMARY =====")

    def mean(x):
        return round(float(np.mean(x)), 4) if len(x) else None

    for k, v in results.items():
        print(f"{k:15s}: {mean(v)}")

    print("\nInterpretation Guide:")
    print("- pixel_iou   : strict pixel overlap")
    print("- dice        : balanced for small lesions")
    print("- tol2/tol3   : clinical tolerance ±2/3 px")
    print("- lesion_iou  : object-level matching")
    print("- positive_iou: only slices with drusen")


if __name__ == "__main__":
    run_inference()
