import cv2
import numpy as np
import json
import sys
from pathlib import Path

# CONFIG
CROP_Y        = 77          
COPPER_H_MIN  = 5 
COPPER_H_MAX  = 50
COPPER_S_MIN  = 50     
COPPER_V_MIN  = 50
MORPH_KERNEL  = 5           
MIN_DEFECT_PX = 300         


# Load & Split
def load_and_split(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    h, w = img.shape[:2]
    mid  = w // 2
    gt_raw  = img[CROP_Y:, :mid]
    cam_raw = img[CROP_Y:, mid:]
    return gt_raw, cam_raw

# Extract GT mask (binary already)

def extract_gt_mask(gt_img):
    gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    _, raw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Remove border artifacts (components spanning full image width/height)
    h, w = raw.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
    clean = np.zeros_like(raw)
    for i in range(1, num):
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if bw >= w - 5 or bh >= h - 5:
            continue   # skip border artifact
        clean[labels == i] = 255
    return clean

# Extract camera copper mask (HSV)
def extract_cam_mask(cam_img):
    hsv = cv2.cvtColor(cam_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        (COPPER_H_MIN, COPPER_S_MIN, COPPER_V_MIN),
        (COPPER_H_MAX, 255,          255)
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL, MORPH_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)   # remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)   # fill glare holes
    return mask

# Register camera mask to GT

def register(gt_mask, cam_mask):
    orb = cv2.ORB_create(500)
    kp1, d1 = orb.detectAndCompute(gt_mask,  None)
    kp2, d2 = orb.detectAndCompute(cam_mask, None)

    if d1 is None or d2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("[WARN] Not enough features for registration — using raw mask")
        return cam_mask

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)

    if len(matches) < 4:
        print("[WARN] Too few matches — using raw mask")
        return cam_mask

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, _  = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    if H is None:
        print("[WARN] Homography failed — using raw mask")
        return cam_mask

    h, w  = gt_mask.shape
    aligned = cv2.warpPerspective(cam_mask, H, (w, h))
    return aligned


# Detect Defects
def detect_defects(gt_mask, cam_mask):
    defects = []

    # Label GT traces
    num_gt, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(gt_mask)

    img_h, img_w = gt_mask.shape

    # BREAK: copper missing from GT region
    for trace_id in range(1, num_gt):
        bw  = int(gt_stats[trace_id, cv2.CC_STAT_WIDTH])
        bh  = int(gt_stats[trace_id, cv2.CC_STAT_HEIGHT])

        # Skip border artifact (spans full image dimensions)
        if bw >= img_w - 5 or bh >= img_h - 5:
            continue

        gt_region   = (gt_labels == trace_id).astype(np.uint8) * 255
        gt_area     = gt_stats[trace_id, cv2.CC_STAT_AREA]

        overlap      = cv2.bitwise_and(gt_region, cam_mask)
        overlap_area = int(np.sum(overlap > 0))
        missing_pct  = 1.0 - overlap_area / max(gt_area, 1)

        # Primary check: did the trace split into 2+ disconnected pieces?
        cam_in_region = cv2.bitwise_and(cam_mask, gt_region)
        n_segs, _     = cv2.connectedComponents(cam_in_region)
        real_segs     = n_segs - 1   # subtract background label

        is_split  = real_segs >= 2
        is_mostly_missing = missing_pct > 0.40

        if is_split or is_mostly_missing:
            x = int(gt_stats[trace_id, cv2.CC_STAT_LEFT])
            y = int(gt_stats[trace_id, cv2.CC_STAT_TOP])
            w = int(gt_stats[trace_id, cv2.CC_STAT_WIDTH])
            h = int(gt_stats[trace_id, cv2.CC_STAT_HEIGHT])

            # Locate the actual gap (centroid of missing copper)
            missing_region = cv2.bitwise_and(gt_region, cv2.bitwise_not(cam_mask))
            M = cv2.moments(missing_region)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            defects.append({
                "type":        "BREAK",
                "trace_id":    trace_id,
                "bbox":        [x, y, w, h],
                "center":      [cx, cy],
                "missing_pct": round(missing_pct * 100, 1),
                "segments":    real_segs,
            })

    # SHORT: extra copper in gap regions 
    gap_mask    = cv2.bitwise_not(gt_mask)
    extra_raw   = cv2.bitwise_and(cam_mask, gap_mask)

    # Erode edges to ignore fringe pixels near trace edges
    k_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    extra   = cv2.morphologyEx(extra_raw, cv2.MORPH_ERODE, k_edge)

    num_ex, ex_labels, ex_stats, _ = cv2.connectedComponentsWithStats(extra)

    for ex_id in range(1, num_ex):
        area = int(ex_stats[ex_id, cv2.CC_STAT_AREA])
        if area < MIN_DEFECT_PX:
            continue

        ex_region = (ex_labels == ex_id).astype(np.uint8) * 255
        dil        = cv2.dilate(ex_region, k_edge, iterations=2)

        # Which GT traces does this extra copper touch?
        touched = set(int(v) for v in gt_labels[dil > 0].flatten()) - {0}

        x = int(ex_stats[ex_id, cv2.CC_STAT_LEFT])
        y = int(ex_stats[ex_id, cv2.CC_STAT_TOP])
        w = int(ex_stats[ex_id, cv2.CC_STAT_WIDTH])
        h_  = int(ex_stats[ex_id, cv2.CC_STAT_HEIGHT])

        M   = cv2.moments(ex_region)
        cx  = int(M["m10"] / M["m00"]) if M["m00"] > 0 else x + w // 2
        cy  = int(M["m01"] / M["m00"]) if M["m00"] > 0 else y + h_ // 2

        defects.append({
            "type":           "SHORT",
            "bbox":           [x, y, w, h_],
            "center":         [cx, cy],
            "area_px":        area,
            "bridges_traces": sorted(touched),
        })

    return defects


# Confidence Score
def confidence_score(gt_mask, cam_mask):
    gt_area     = int(np.sum(gt_mask  > 0))
    cam_area    = int(np.sum(cam_mask > 0))
    correct     = int(np.sum(cv2.bitwise_and(gt_mask, cam_mask) > 0))
    missing     = int(np.sum(cv2.bitwise_and(gt_mask,  cv2.bitwise_not(cam_mask)) > 0))
    extra       = int(np.sum(cv2.bitwise_and(cam_mask, cv2.bitwise_not(gt_mask))  > 0))

    precision   = correct / max(correct + extra,   1)
    recall      = correct / max(correct + missing, 1)
    f1          = 2 * precision * recall / max(precision + recall, 1e-9)
    iou         = correct / max(correct + missing + extra, 1)

    return {
        "precision":         round(precision * 100, 1),
        "recall":            round(recall    * 100, 1),
        "f1_score":          round(f1        * 100, 1),
        "iou":               round(iou       * 100, 1),
        "gt_copper_px":      gt_area,
        "cam_copper_px":     cam_area,
        "correct_px":        correct,
        "missing_px":        missing,
        "extra_px":          extra,
    }


# Render Output Images
def render_clean_copper(gt_mask, size):
    """Render idealized copper board from GT mask — no glare, no noise."""
    h, w = size
    out  = np.zeros((h, w, 3), dtype=np.uint8)
    # PCB substrate color: dark green
    out[:] = (30, 60, 30)
    # Copper color: warm orange
    out[gt_mask > 0] = (30, 140, 210)   # BGR: orange
    # Subtle edge glow
    edges = cv2.Canny(gt_mask, 50, 150)
    edges = cv2.dilate(edges, np.ones((2,2)))
    out[edges > 0] = (60, 180, 255)     # brighter orange at edges
    return out

def render_annotated(cam_img, cam_mask, gt_mask, defects):
    """Overlay defect annotations on camera image."""
    out = cam_img.copy()

    # Dim the image slightly for annotation visibility
    out = cv2.convertScaleAbs(out, alpha=0.85, beta=0)

    # Draw copper overlay (semi-transparent green tint for correct copper)
    correct_mask = cv2.bitwise_and(gt_mask, cam_mask)
    correct_overlay = out.copy()
    correct_overlay[correct_mask > 0] = [50, 200, 50]
    out = cv2.addWeighted(out, 0.8, correct_overlay, 0.2, 0)

    for d in defects:
        x, y, w, h = d["bbox"]
        cx, cy     = d["center"]

        if d["type"] == "BREAK":
            color = (0, 0, 255)      # Red for break
            label = f"BREAK ({d['missing_pct']}% missing)"
            # Draw crosshair at gap location
            cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 30, 2)
        else:
            color = (0, 165, 255)    # Orange for short
            label = f"SHORT (bridges {d['bridges_traces']})"
            cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_TILTED_CROSS, 30, 2)

        # Bounding box
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx = max(0, min(x, out.shape[1] - tw - 4))
        ly = max(th + 4, y - 4)
        cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 4, ly), color, -1)
        cv2.putText(out, label, (lx + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return out

def render_diff(gt_mask, cam_mask):
    """Show what's missing and what's extra."""
    h, w = gt_mask.shape
    out  = np.zeros((h, w, 3), dtype=np.uint8)
    out[:] = (20, 20, 20)   # dark background

    correct  = cv2.bitwise_and(gt_mask, cam_mask)
    missing  = cv2.bitwise_and(gt_mask,  cv2.bitwise_not(cam_mask))
    extra    = cv2.bitwise_and(cam_mask, cv2.bitwise_not(gt_mask))

    out[correct > 0] = (50, 200, 50)    # green  = correct copper
    out[missing > 0] = (0,   0, 255)    # red    = missing (breaks)
    out[extra   > 0] = (0, 165, 255)    # orange = extra (shorts)
    return out





def run(input_path, out_dir=None):
    if out_dir is None:
        out_dir = str(Path(__file__).parent / "outputs")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("Loading image...")
    gt_raw, cam_raw = load_and_split(input_path)

    print("Extracting masks...")
    gt_mask  = extract_gt_mask(gt_raw)
    cam_mask = extract_cam_mask(cam_raw)

    print("Registering camera to GT...")
    cam_aligned = register(gt_mask, cam_mask)

    print("Detecting defects...")
    defects = detect_defects(gt_mask, cam_aligned)

    print("Computing confidence score...")
    score = confidence_score(gt_mask, cam_aligned)

    print("Rendering output images...")
    h, w   = gt_mask.shape
    clean  = render_clean_copper(gt_mask, (h, w))
    annot  = render_annotated(cam_raw, cam_aligned, gt_mask, defects)
    diff   = render_diff(gt_mask, cam_aligned)


    pad    = 10
    label_h = 30

    def add_label(img, text, color=(255,255,255)):
        labeled = np.zeros((label_h + img.shape[0], img.shape[1], 3), np.uint8)
        labeled[label_h:] = img
        cv2.putText(labeled, text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
        return labeled

    clean_labeled = add_label(clean, "CLEAN RECONSTRUCTION (Ground Truth)", (50,220,50))
    annot_labeled = add_label(annot, "DETECTED DEFECTS", (80,80,255))
    diff_labeled  = add_label(diff,  "DIFF MAP  (green=ok  red=break  orange=short)")


    legend = np.zeros((diff.shape[0] + label_h, diff.shape[1], 3), np.uint8)
    legend[:] = (15, 15, 15)
    breaks_  = [d for d in defects if d["type"] == "BREAK"]
    shorts_  = [d for d in defects if d["type"] == "SHORT"]

    lines = [
        ("DETECTION RESULTS", (200,200,200)),
        ("", (0,0,0)),
        (f"  Defects found:  {len(defects)}", (200,200,200)),
        (f"  Breaks:         {len(breaks_)}",  (80, 80,255)),
        (f"  Shorts:         {len(shorts_)}",  (80,165,255)),
        ("", (0,0,0)),
        ("CONFIDENCE SCORES", (200,200,200)),
        (f"  Precision:  {score['precision']}%", (100,220,100)),
        (f"  Recall:     {score['recall']}%",    (100,220,100)),
        (f"  F1 Score:   {score['f1_score']}%",  (100,220,100)),
        (f"  IoU:        {score['iou']}%",        (100,220,100)),
    ]

    for b in breaks_:
        lines.append((f"  [BREAK] trace#{b['trace_id']} — {b['missing_pct']}% missing", (80,80,255)))
    for s in shorts_:
        lines.append((f"  [SHORT] {s['area_px']}px bridges {s['bridges_traces']}", (80,165,255)))

    for i, (text, color) in enumerate(lines):
        y_pos = label_h + 20 + i * 22
        if y_pos < legend.shape[0] - 10:
            cv2.putText(legend, text, (8, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    row1  = np.hstack([clean_labeled, np.full((clean_labeled.shape[0], pad, 3), 30, np.uint8), annot_labeled])
    row2  = np.hstack([diff_labeled,  np.full((diff_labeled.shape[0],  pad, 3), 30, np.uint8), legend])
    final = np.vstack([row1, np.full((pad, row1.shape[1], 3), 30, np.uint8), row2])

    out_img_path = f"{out_dir}/pcb_analysis.png"
    cv2.imwrite(out_img_path, final)
    print(f"Saved → {out_img_path}")

    # JSON results 
    results = {"defects": defects, "confidence": score}
    out_json_path = f"{out_dir}/pcb_results.json"
    with open(out_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out_json_path}")

    print("\nRESULTS")
    print(f"Breaks found:  {len(breaks_)}")
    print(f"Shorts found:  {len(shorts_)}")
    print(f"F1 Score:      {score['f1_score']}%")
    print(f"IoU:           {score['iou']}%")
    print(f"Precision:     {score['precision']}%")
    print(f"Recall:        {score['recall']}%")
    for d in defects:
        print(f"  → {d}")
    print("\n")

    return results

if __name__ == "__main__":
    default_img = str(Path(__file__).parent / "images" / "test 1.png")
    path = sys.argv[1] if len(sys.argv) > 1 else default_img
    run(path)
