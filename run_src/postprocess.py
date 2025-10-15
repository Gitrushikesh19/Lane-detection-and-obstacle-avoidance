import cv2
import numpy as np
from run_src.utils import get_bev_homography, warp_perspective
from typing import List
from torchvision import transforms
import torch

Device = "cuda" if torch.cuda.is_available() else "cpu"
Input_size = (256, 512)
# Save_dir = "/outputs/inference"

transform_img = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(Input_size),
    transforms.ToTensor()
])

def predict_mask(model, frame: np.ndarray, device=Device) -> np.ndarray:
    #Returns row probability mask
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #Input
    inp = transform_img(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp)
    pred_np = pred.squeeze().cpu().numpy() #(H, W) float in [0, 1]
    pred_resized = cv2.resize(pred_np, (frame.shape[1], frame.shape[0])) #Back to original size
    return pred_resized

def draw_lane_overlay(frame, lane_poly, center_pts, color=(0, 255, 0), alpha=0.4, Minv=None):
    overlay = frame.copy()

    if Minv is not None:
        if lane_poly is not None:
            pts = cv2.perspectiveTransform(lane_poly.reshape(-1, 1, 2).astype(np.float32), Minv)
            lane_poly = pts.reshape(-1, 2)

        if center_pts is not None:
            pts = cv2.perspectiveTransform(center_pts.reshape(-1, 1, 2).astype(np.float32), Minv)
            center_pts = pts.reshape(-1, 2).astype(int)


    if lane_poly is not None and lane_poly.shape[0] > 2:
        cv2.fillPoly(overlay, [lane_poly.astype(np.int32)], color)
        #Draw centerline
    if center_pts is not None:
        for i in range(len(center_pts)-1):
            cv2.line(overlay, tuple(center_pts[i]), tuple(center_pts[i+1]), (0, 0, 255), 2)
    polygon_line_combined = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
    return polygon_line_combined

def clean_mask(mask):
    mask = (mask > 0.5).astype(np.uint8)*255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def process_lane_mask(mask):
    h, w = mask.shape[:2]
# BEV homography matrics
    H, Minv = get_bev_homography((h, w))
# Warp mask into BEV
    bev_mask = warp_perspective(mask, H, (w, h))
    return bev_mask

def split_lane_components(binary_mask: np.ndarray, max_components=2) -> List[np.ndarray]:
    num_labels, labels_idx, stats, centroids = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    comps = []
    for label in range(1, num_labels):
        comp_mask = (labels_idx == label).astype(np.uint8) * 255
        comps.append((comp_mask, centroids[label]))

    if not comps:
        return []

    else:
        comps = sorted(comps, key=lambda x: x[1][0])
        comps = [c[0] for c in comps[:max_components]]
        return comps

def fit_poly_to_mask(mask: np.ndarray, order=2):
    y_idx, x_idx = np.nonzero(mask)
    if len(x_idx) < 50:
        return None

    coeffs = np.polyfit(y_idx, x_idx, order) #Gives coeffs for polyval
    return coeffs

def eval_poly(coeffs, y_vals):
    return np.polyval(coeffs, y_vals)

def lane_polygon_from_two_fits(left_fit, right_fit, img_h, img_w):
    y_vals = np.linspace(0, img_h-1, img_h).astype(np.int32)
    left_x = np.clip(eval_poly(left_fit, y_vals).astype(np.int32), 0, img_w-1)
    right_x = np.clip(eval_poly(right_fit, y_vals).astype(np.int32), 0, img_w-1)
    left_pts = np.stack([left_x, y_vals], axis=1)
    right_pts = np.stack([right_x, y_vals], axis=1)
    poly = np.vstack([left_pts, right_pts[::-1]])
    return poly

def compute_centerline(left_fit, right_fit, img_h):
    y_vals = np.linspace(0, img_h-1, img_h)
    left_x = eval_poly(left_fit, y_vals)
    right_x = eval_poly(right_fit, y_vals)
    center_x = (left_x +right_x) / 2
    pts = np.stack([center_x.astype(np.int32), y_vals.astype(np.int32)], axis=1)
    return pts

def compute_lateral_offset(centerline_pts, img_center_x, xm_per_pix):
    bottom_x = centerline_pts[-1, 0]
    offset_m = (bottom_x - img_center_x)*xm_per_pix
    return offset_m
