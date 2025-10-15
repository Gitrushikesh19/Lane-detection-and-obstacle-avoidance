import numpy as np
from scipy.spatial.distance import directed_hausdorff

def iou_mask(pred_mask, gt_mask):
    p = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return None
    return float(inter) / (union + 1e-8)

def mean_lateral_error(pred_center_pts, gt_center_pts, xm_per_pix):
    if pred_center_pts is None or gt_center_pts is None:
        return None
    # ensure sorted by y
    pred = pred_center_pts[np.argsort(pred_center_pts[:,1])]
    gt   = gt_center_pts[np.argsort(gt_center_pts[:,1])]
    # find overlapping y-range
    y_min = max(pred[:,1].min(), gt[:,1].min())
    y_max = min(pred[:,1].max(), gt[:,1].max())
    if y_max <= y_min:
        return None
    y_vals = np.arange(int(y_min), int(y_max)+1)
    pred_x = np.interp(y_vals, pred[:,1], pred[:,0])
    gt_x   = np.interp(y_vals, gt[:,1],   gt[:,0])
    err_pixels = np.abs(pred_x - gt_x)
    return float(np.mean(err_pixels)) * xm_per_pix
