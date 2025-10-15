import cv2
import numpy as np
from typing import Tuple
import json
from pathlib import Path
import torch
from run_src.model_unet import UNet
from run_src.postprocess import process_lane_mask, clean_mask, split_lane_components, \
    fit_poly_to_mask, lane_polygon_from_two_fits, compute_centerline

points = []
window_name = "Click 4 points in the order (top_left, top_right, bottom_right, bottom_left)"

Device = "cuda" if torch.cuda.is_available() else "cpu"
Weights = "/outputs/checkpoints/unet_best.pth"
Input_size = (256, 512)
# Save_dir = "/outputs/inference"

def dice_score(y_pred, y_true, smooth=0.0001):
    y_pred = (y_pred > 0.5).float()

    #Flatten these tensors
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    intersection = (y_pred*y_true).sum()
    dice = (2*intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

    return dice

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point selected: {(x, y)}")
        if len(points) == 4:
            cv2.destroyAllWindows()

def select_src_points(imgs, save_path="homography_points.json"):
    global points
    points = []
    src_frame = imgs.copy()
    cv2.imshow(window_name, src_frame)
    cv2.setMouseCallback(window_name, click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 4:
        raise ValueError("You must select 4 points")

    with open(save_path, "w") as file:
        json.dump(points, file)

    return np.float32(points)

def load_src_points(load_path="homography_points.json"):
    with open(load_path, "r") as file:
        points = json.load(file)
    return np.float32(points)

def get_bev_homography(frame, src_points=None, save_path="homography_points.json") -> Tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]

    # src = np.float32([
    #     [int(0.45*w), int(0.6*h)], #Top_left
    #     [int(0.55*w), int(0.6*h)], #Top_right
    #     [int(0.95*w), int(0.95*h)], #Bottom_right
    #     [int(0.05*w), int(0.95*h)] #Bottom_left
    # ])

    if src_points is None:
        if Path(save_path).exists():
            src_points = load_src_points(save_path)
        else:
            src_points = select_src_points(frame, save_path)

    bev_w, bev_h = w, h
    destination_rect = np.float32([
        [int(0.2*bev_w), 0],
        [int(0.8*bev_w), 0],
        [int(0.8*bev_w), bev_h],
        [int(0.2*bev_w), bev_h]
    ])

    H = cv2.getPerspectiveTransform(src_points, destination_rect)
    Minv = cv2.getPerspectiveTransform(destination_rect, src_points)

    return H, Minv

def warp_perspective(img: np.ndarray, H: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(img, H, out_size, flags=cv2.INTER_LINEAR)

def pixel_to_meters_conversion():
    xm_per_pix = 2 / 512
    ym_per_pix = 20 / 640
    return xm_per_pix, ym_per_pix

def load_model(weights_path, device="cpu", state_key_names=None, strict=True):

    device = torch.device(device)
    model = UNet(n_channels=3, n_classes=1)   # keep model on CPU initially

    ckpt = torch.load(str(weights_path), map_location="cpu")

    # Determine where the actual state_dict is
    state_dict = None
    if isinstance(ckpt, dict):
        # common keys to check if user passed a full checkpoint:
        candidates = state_key_names or ['model_state', 'state_dict', 'state', 'optimizer']
        for k in candidates:
            if k in ckpt:
                state_dict = ckpt[k]
                break
        if state_dict is None:
            # maybe the checkpoint is already a raw state_dict (no extra keys)
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state_dict = ckpt
            else:
                raise RuntimeError("Checkpoint format not recognized. Keys: " + ", ".join(ckpt.keys()))
    else:
        # ckpt is likely a raw state dict
        state_dict = ckpt

    load_res = model.load_state_dict(state_dict, strict=strict)
    missing = getattr(load_res, 'missing_keys', None)
    unexpected = getattr(load_res, 'unexpected_keys', None)
    if missing:
        print("[load_model] Missing keys:", missing)
    if unexpected:
        print("[load_model] Unexpected keys:", unexpected)

    model.to(device)
    model.eval()
    return model

def load_lane_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def polyline_to_mask(polyline, img_shape, half_width=6):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polyline, dtype=np.int32).reshape((-1,1,2))
    # draw thick polyline
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=half_width*2, lineType=cv2.LINE_AA)
    # Optionally, to ensure solid mask with no anti-alias:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def json_polylines_to_mask(lane_obj, img_shape, thickness=6):
    lanes = lane_obj["lanes"]
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(lanes) >= 2:
        bottoms = []
        bottom_y = h - 1
        for l in lanes:
            pts = np.array(l["polyline"])
            ys = pts[:,1]; xs = pts[:,0]
            valid = (ys >= 0) & (ys <= bottom_y)
            if valid.sum() < 2:
                bottoms.append(float(xs[0]))
            else:
                bottoms.append(float(np.interp(bottom_y, ys[::-1], xs[::-1])))
        sorted_idx = np.argsort(bottoms)
        left = np.array(lanes[sorted_idx[0]]["polyline"])
        right = np.array(lanes[sorted_idx[-1]]["polyline"])
        y_vals = np.linspace(0, h - 1, h)
        left_x = np.interp(y_vals, left[:,1][::-1], left[:,0][::-1])
        right_x = np.interp(y_vals, right[:,1][::-1], right[:,0][::-1])
        poly = np.vstack([np.stack([left_x, y_vals], axis=1),
                          np.stack([right_x[::-1], y_vals[::-1]], axis=1)])
        poly = np.round(poly).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)
    else:
        for l in lanes:
            m = polyline_to_mask(l["polyline"], img_shape, half_width=thickness)
            mask = cv2.bitwise_or(mask, m)
    return mask

def centerline_corridor_from_mask(mask, bev_size):
    bev_h, bev_w = bev_size
    binary = clean_mask(mask)
    bev_binary = process_lane_mask(binary)

    comps = split_lane_components(bev_binary, max_components=2)
    left_fit, right_fit = None, None

    if len(comps) == 1:
        part = comps[0]
        fit = fit_poly_to_mask(part)
        left_fit = fit
        right_fit = fit
    elif len(comps) >= 2:
        left_fit = fit_poly_to_mask(comps[0])
        right_fit = fit_poly_to_mask(comps[1])

    if left_fit is not None and right_fit is not None:
        lane_poly = lane_polygon_from_two_fits(left_fit, right_fit, bev_h, bev_w)
        center_pts_bev = compute_centerline(left_fit, right_fit, bev_h)
        # create corridor mask in bev
        corridor_mask_bev = np.zeros((bev_h, bev_w), dtype=np.uint8)
        try:
            cv2.fillPoly(corridor_mask_bev, [lane_poly], 255)
        except:
            corridor_mask_bev = None
        return corridor_mask_bev, center_pts_bev
