import os
import cv2
import torch
from run_src.utils import get_bev_homography, pixel_to_meters_conversion, load_model, \
    load_lane_json, json_polylines_to_mask, centerline_corridor_from_mask
from run_src.eval_metrics import mean_lateral_error, iou_mask
from run_src.postprocess import predict_mask, clean_mask


IMG_DIR = "data/raw/images"
JSON_DIR = "data/annotations/lanes_json"
MASK_DIR = "data/raw/masks"
Device = "cuda" if torch.cuda.is_available() else "cpu"
Weights = "/outputs/checkpoints/unet_best.pth"
MODEL = load_model(Weights, device=Device)

xm_per_pix, ym_per_pix = pixel_to_meters_conversion()

iou_pixels = []
iou_bev_corridor = []
lat_errs = []

for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.lower().endswith((".jpg",".png")):
        continue
    img_path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(img_path)
    h,w = img.shape[:2]

    # 1) Predicted mask -> binarize
    prob = predict_mask(MODEL, img, device=Device)
    pred_mask_cam = clean_mask(prob)   # 0/255 in camera coords

    # 2) Predicted centerline & corridor in BEV
    H, Minv = get_bev_homography(img)    # camera->BEV
    bev_size = (h, w)
    pred_center_bev, pred_corridor_bev = centerline_corridor_from_mask(pred_mask_cam, H, bev_size)

    # 3) GT: from JSON (or load GT mask if available)
    json_path = os.path.join(JSON_DIR, os.path.splitext(fname)[0] + ".json")
    if os.path.exists(json_path):
        js = load_lane_json(json_path)
        gt_mask_cam = json_polylines_to_mask(js, (h,w), thickness=6)
    else:
        gt_mask_cam = cv2.imread(os.path.join(MASK_DIR, os.path.splitext(fname)[0]+".png"), cv2.IMREAD_GRAYSCALE)

    gt_center_bev, gt_corridor_bev = centerline_corridor_from_mask(gt_mask_cam, H, bev_size)

    # 4) Metrics
    pixel_iou = iou_mask(pred_mask_cam, gt_mask_cam)
    corridor_iou = iou_mask(pred_corridor_bev, gt_corridor_bev)
    lat_err = mean_lateral_error(pred_center_bev, gt_center_bev, xm_per_pix)

    iou_pixels.append(pixel_iou)
    iou_bev_corridor.append(corridor_iou)
    lat_errs.append(lat_err)
