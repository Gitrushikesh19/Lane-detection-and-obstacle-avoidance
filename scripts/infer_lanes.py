import cv2
from run_src.utils import get_bev_homography, pixel_to_meters_conversion, load_model
from run_src.postprocess import clean_mask, split_lane_components, fit_poly_to_mask, lane_polygon_from_two_fits, \
    compute_centerline, compute_lateral_offset, process_lane_mask, predict_mask, draw_lane_overlay
from pathlib import Path
import torch
import time

weights = "/outputs/checkpoints/unet_best.pth"
Device = "cuda" if torch.cuda.is_available() else "cpu"
Input_size = (256, 512)
# Save_dir = "/outputs/inference"

def infer_on_stream(source=0, save_path=None, visualize=True, max_frames=None):
    model = load_model(weights, Device)
    cap = None
    files = []
    is_folder = False

    if isinstance(source, (str, Path)) and Path(source).is_dir():
        is_folder = True
        files = sorted([str(p) for p in Path(source).glob("*") if p.suffix.lower() in [".jpg", ".png"]])
        total = len(files)
    else:
        cap = cv2.VideoCapture(source)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    # writer = None
    # out_fps = 20

    #Compute BEV homography on the first frame size
    first_frame = True

    if is_folder:
        first_frame = cv2.imread(files[0])
    else:
        #Read one frame to get shape
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Can't read first frame")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) #Reset

    H, Minv = get_bev_homography(first_frame)
    img_h, img_w = first_frame.shape[:2]
    img_center_x = img_w//2
    xm_per_pix, ym_per_pix = pixel_to_meters_conversion()

    frame_idx = 0
    t0 = time.time()
    visibility = None

    while True:
        if is_folder:
            if frame_idx >= total:
                break
            frame_path = files[frame_idx]
            frame = cv2.imread(frame_path)
        else:
            ret, frame = cap.read()
            if not ret:
                break

        prob_mask = predict_mask(model, frame)
        binary = clean_mask(prob_mask)
        #Should provide mask_bev on the below function(comps) & change max_components as per the lane requirements(no. of lanes)
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

        lane_poly = None
        center_pts = None
        offset_m = None

        if left_fit is not None and right_fit is not None:
            lane_poly = lane_polygon_from_two_fits(left_fit, right_fit, img_h, img_w)
            center_pts = compute_centerline(left_fit, right_fit, img_h)
            offset_m = compute_lateral_offset(center_pts, img_center_x, xm_per_pix)

        visible = frame.copy()
        if lane_poly is not None and center_pts is not None:
            #What to write or show here visible while camera output(visible or visibility)
            visibility = draw_lane_overlay(visible, lane_poly, center_pts, Minv=Minv)

        #Annotate offset
        if offset_m is not None:
            text = f"Offset: {offset_m:.2f} m"
            cv2.putText(visible, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        elapsed_time = time.time() - t0
        fps = frame_idx / elapsed_time if elapsed_time > 0 else 0.0
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(visible, fps_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if visualize:
            cv2.imshow("Lane Inference", visibility)
            if cv2.waitKey(1) & 0xff == 27:
                break

        # if save_path:
        #     if writer is None:
        #         # create writer with same shape
        #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        #         writer = cv2.VideoWriter(save_path, fourcc, out_fps, (img_w, img_h))
        #     writer.write(visible)
        #
        # frame_idx += 1
        # if max_frames and frame_idx >= max_frames:
        #     break

    # if writer:
    #     writer.release()
    if cap:
        cap.release()
    cv2.destroyAllWindows()
