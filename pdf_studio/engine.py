import os
import json
import cv2
import numpy as np
import subprocess
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REALESRGAN_EXE = os.path.join(BASE_DIR, "bin", "realesrgan", "realesrgan-ncnn-vulkan.exe")


def is_white_mask(img_bgr, l_thresh=235, c_thresh=16):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    bg_l = cv2.morphologyEx(l, cv2.MORPH_DILATE, k)
    bg_a = cv2.GaussianBlur(a, (35, 35), 0)
    bg_b = cv2.GaussianBlur(b, (35, 35), 0)
    diff_l = bg_l.astype(np.float32) - l.astype(np.float32)
    diff_c = np.sqrt((a.astype(np.float32) - bg_a.astype(np.float32)) ** 2 + (b.astype(np.float32) - bg_b.astype(np.float32)) ** 2)
    return (diff_l <= 20.0) & (diff_c <= 16.0) & (l >= 185)


def estimate_local_background(img_bgr):
    """Estimate smooth local paper background BGR using multi-scale median filtering (eliminates white dilation halos on tinted pages)."""
    h, w = img_bgr.shape[:2]
    sw = max(16, w // 4)
    sh = max(16, h // 4)
    small = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    med = cv2.medianBlur(small, 21)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(med, cv2.MORPH_CLOSE, k)
    blurred = cv2.GaussianBlur(closed, (9, 9), 0)
    return cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)


def pre_clean_page(img_bgr):
    """Clean off-white JPEG noise in paper background and strengthen faint dots/diacritics."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    bg_l = cv2.morphologyEx(l, cv2.MORPH_DILATE, kernel)
    bg_l = cv2.GaussianBlur(bg_l, (21, 21), 0)

    l_float = l.astype(np.float32)
    bg_float = np.maximum(bg_l.astype(np.float32), 200.0)
    l_norm = np.clip((l_float / bg_float) * 255.0, 0, 255)

    norm_01 = l_norm / 255.0
    white_mask = np.clip((norm_01 - 0.91) / 0.07, 0.0, 1.0)
    dark_mask = np.clip((0.68 - norm_01) / 0.68, 0.0, 1.0)

    l_enhanced = l_norm * (1.0 - 0.26 * dark_mask)
    l_enhanced = l_enhanced * (1.0 - white_mask) + 255.0 * white_mask
    l_enhanced = np.clip(l_enhanced, 0, 255).astype(np.uint8)

    chroma_dist = np.sqrt((a.astype(np.float32) - 128.0) ** 2 + (b.astype(np.float32) - 128.0) ** 2)
    paper_mask = ((l_enhanced > 245) & (chroma_dist < 8.0)).astype(np.float32)
    paper_mask = cv2.GaussianBlur(paper_mask, (5, 5), 0)
    a_clean = (a.astype(np.float32) * (1.0 - paper_mask) + 128.0 * paper_mask).astype(np.uint8)
    b_clean = (b.astype(np.float32) * (1.0 - paper_mask) + 128.0 * paper_mask).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([l_enhanced, a_clean, b_clean]), cv2.COLOR_LAB2BGR)


def enhance_page_resolution(img_bgr, scale=3, use_gpu=True):
    """Super-resolve a low-res page image using Real-ESRGAN + dot/diacritic-preserving hybrid restoration."""
    cleaned = pre_clean_page(img_bgr)
    h, w = cleaned.shape[:2]
    lanczos = cv2.resize(cleaned, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
    gauss = cv2.GaussianBlur(lanczos, (0, 0), sigmaX=1.2)
    lanczos_sharp = cv2.addWeighted(lanczos, 1.5, gauss, -0.5, 0)

    if use_gpu and os.path.exists(REALESRGAN_EXE) and max(h, w) < 1400:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "in.png")
            out_path = os.path.join(tmpdir, "out.png")
            cv2.imwrite(in_path, cleaned)
            try:
                subprocess.run(
                    [REALESRGAN_EXE, "-i", in_path, "-o", out_path, "-n", "realesr-animevideov3", "-s", str(scale)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                sr_bgr = cv2.imread(out_path)
                if sr_bgr is not None and sr_bgr.shape[:2] == lanczos_sharp.shape[:2]:
                    blended = cv2.addWeighted(sr_bgr, 0.72, lanczos_sharp, 0.28, 0)
                    dark_protect = np.minimum(blended, cv2.addWeighted(lanczos_sharp, 0.85, blended, 0.15, 10))
                    g2 = cv2.GaussianBlur(dark_protect, (0, 0), sigmaX=0.85)
                    return cv2.addWeighted(dark_protect, 1.28, g2, -0.28, 0)
            except Exception:
                pass

    return lanczos_sharp


def compact_white_rows_both(patch_bgr, patch_ink, ratio_remove, min_keep_run=8):
    if ratio_remove <= 0.0:
        return patch_bgr, patch_ink
    h, w = patch_ink.shape
    row_has_ink = np.sum(patch_ink, axis=1) > 0

    keep_row = np.ones(h, dtype=bool)
    runs = []
    in_run = False
    start = 0
    for y in range(h):
        if not row_has_ink[y]:
            if not in_run:
                in_run = True
                start = y
        else:
            if in_run:
                in_run = False
                runs.append((start, y))
    if in_run:
        runs.append((start, h))

    for (r0, r1) in runs:
        if r0 == 0 or r1 == h:
            continue
        run_len = r1 - r0
        if run_len > min_keep_run:
            max_rem = min(int(run_len * 0.68), run_len - min_keep_run)
            can_rem = int(round(max_rem * min(1.0, ratio_remove)))
            if can_rem > 0:
                mid = (r0 + r1) // 2
                rem_start = mid - can_rem // 2
                keep_row[rem_start : rem_start + can_rem] = False

    return patch_bgr[keep_row, :, :], patch_ink[keep_row, :]


def compact_white_cols_both(patch_bgr, patch_ink, ratio_remove, min_col_gap=45):
    """Compact wide empty vertical strips between columns without touching normal word spaces (< min_col_gap)."""
    if ratio_remove <= 0.0:
        return patch_bgr, patch_ink
    h, w = patch_ink.shape
    col_has_ink = np.sum(patch_ink > 0, axis=0) > 0

    keep_col = np.ones(w, dtype=bool)
    runs = []
    in_run = False
    start = 0
    for x in range(w):
        if not col_has_ink[x]:
            if not in_run:
                in_run = True
                start = x
        else:
            if in_run:
                in_run = False
                runs.append((start, x))
    if in_run:
        runs.append((start, w))

    for (c0, c1) in runs:
        if c0 == 0 or c1 == w:
            continue
        run_len = c1 - c0
        if run_len >= min_col_gap:
            max_rem = min(int(run_len * 0.78), run_len - min_col_gap // 2)
            can_rem = int(round(max_rem * min(1.0, ratio_remove)))
            if can_rem > 0:
                mid = (c0 + c1) // 2
                rem_start = mid - can_rem // 2
                keep_col[rem_start : rem_start + can_rem] = False

    return patch_bgr[:, keep_col, :], patch_ink[:, keep_col]


def segment_page_masks(img_bgr):
    """Segment page into obstacle_mask, obs_dilated, text_ink_mask, and content bounds."""
    H, W = img_bgr.shape[:2]
    white = is_white_mask(img_bgr)
    ink = (~white).astype(np.uint8)

    n0, l0, s0, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for idx in range(1, n0):
        if s0[idx, cv2.CC_STAT_AREA] < 10:
            ink[l0 == idx] = 0

    ink_d = cv2.dilate(ink, np.ones((3, 3), np.uint8), iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_d, connectivity=8)

    obstacle_mask = np.zeros((H, W), dtype=np.uint8)
    text_ink_mask = np.zeros((H, W), dtype=np.uint8)

    y_top_safe = int(H * 0.045)
    y_bot_safe = int(H * 0.925)
    x_left_safe = int(W * 0.045)
    x_right_safe = int(W * 0.955)

    content_y1, content_y2 = int(H * 0.15), int(H * 0.85)
    content_x1, content_x2 = int(W * 0.15), int(W * 0.85)

    for lbl in range(1, num_labels):
        x, y, w, h, area = stats[lbl]
        touches_bleed = (x <= 5) or (x + w >= W - 5) or (y <= 5) or (y + h >= H - 5)
        if not touches_bleed and area > 25:
            content_x1 = min(content_x1, x)
            content_y1 = min(content_y1, y)
            content_x2 = max(content_x2, x + w)
            content_y2 = max(content_y2, y + h)

        in_page_num_badge = (y > int(H * 0.85)) and (h > int(H * 0.045)) and (area > int(H * W * 0.0025))
        outside_safe = (y < y_top_safe) or (y + h > y_bot_safe) or (x < x_left_safe) or (x + w > x_right_safe)
        is_illustration = (
            (h > int(H * 0.055))
            or (w > int(W * 0.48))
            or (w * h > int(H * W * 0.015))
            or (area > int(H * W * 0.0030))
        )

        if outside_safe or in_page_num_badge or is_illustration:
            obstacle_mask[labels == lbl] = 1
        else:
            text_ink_mask[labels == lbl] = 1

    # Also detect horizontal dotted/dashed separator rules and treat them as obstacles
    horiz_close = cv2.morphologyEx(text_ink_mask, cv2.MORPH_CLOSE, np.ones((1, 25), np.uint8))
    nh, lh, sth, _ = cv2.connectedComponentsWithStats(horiz_close, connectivity=8)
    for idx in range(1, nh):
        hx, hy, hw, hh, _ = sth[idx]
        if hw > int(W * 0.40) and hh < int(H * 0.012):
            obstacle_mask[(lh == idx) & (text_ink_mask > 0)] = 1
            text_ink_mask[lh == idx] = 0

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(W * 0.025), int(H * 0.025)))
    obs_closed = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, close_k)
    obs_dilated = cv2.dilate(obs_closed, np.ones((int(H * 0.012), int(W * 0.012)), np.uint8), iterations=1)
    text_ink_mask[obs_dilated > 0] = 0
    return obs_closed, obs_dilated, text_ink_mask, (content_x1, content_y1, content_x2, content_y2)


def detect_text_blocks(img_bgr):
    """Detect 2D text blocks and calculate each block's recommended auto-fit scale and position."""
    H, W = img_bgr.shape[:2]
    obstacle_mask, obs_dilated, text_ink_mask, _ = segment_page_masks(img_bgr)
    bg_bgr = estimate_local_background(img_bgr)

    block_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * 0.22), int(H * 0.032)))
    block_blobs = cv2.dilate(text_ink_mask, block_kernel, iterations=1)
    block_blobs[obs_dilated > 0] = 0

    n_blk, blk_labels, blk_stats, _ = cv2.connectedComponentsWithStats(block_blobs, connectivity=8)
    raw_blocks = []
    for b in range(1, n_blk):
        bx, by, bw, bh, barea = blk_stats[b]
        sub_ink = np.where((text_ink_mask[by : by + bh, bx : bx + bw] > 0) & (blk_labels[by : by + bh, bx : bx + bw] == b))
        if len(sub_ink[0]) < 35:
            continue
        ty1 = by + int(np.min(sub_ink[0]))
        ty2 = by + int(np.max(sub_ink[0])) + 1
        tx1 = bx + int(np.min(sub_ink[1]))
        tx2 = bx + int(np.max(sub_ink[1])) + 1
        if (tx2 - tx1) >= int(W * 0.05) and (ty2 - ty1) >= int(H * 0.008):
            raw_blocks.append([tx1, ty1, tx2, ty2])

    raw_blocks.sort(key=lambda b: b[1])

    blocks = []
    for rb in raw_blocks:
        if not blocks:
            blocks.append(rb)
        else:
            prev = blocks[-1]
            gap_y = rb[1] - prev[3]
            ox1 = max(prev[0], rb[0])
            ox2 = min(prev[2], rb[2])
            if ox2 <= ox1:
                ox1, ox2 = min(prev[0], rb[0]), max(prev[2], rb[2])
            yc1 = max(0, min(prev[3], rb[1]))
            yc2 = min(H - 1, max(prev[3], rb[1]))
            if gap_y <= int(H * 0.035) and np.sum(obstacle_mask[yc1 : yc2 + 1, ox1:ox2]) == 0:
                prev[0] = min(prev[0], rb[0])
                prev[1] = min(prev[1], rb[1])
                prev[2] = max(prev[2], rb[2])
                prev[3] = max(prev[3], rb[3])
            else:
                blocks.append(rb)

    pad_src = max(6, int(H * 0.004))
    safety_px = max(8, int(H * 0.005))
    min_keep = max(6, int(H * 0.004))
    y_top_safe = int(H * 0.045)
    y_bot_safe = int(H * 0.925)
    x_left_safe = int(W * 0.045)
    x_right_safe = int(W * 0.955)

    detected = []
    for idx, blk in enumerate(blocks):
        bx1 = max(0, blk[0] - pad_src)
        by1 = max(0, blk[1] - pad_src)
        bx2 = min(W, blk[2] + pad_src)
        by2 = min(H, blk[3] + pad_src)

        patch_bgr = img_bgr[by1:by2, bx1:bx2].copy()
        patch_bg = bg_bgr[by1:by2, bx1:bx2]
        patch_ink = text_ink_mask[by1:by2, bx1:bx2].copy()
        patch_obs = obs_dilated[by1:by2, bx1:bx2] > 0
        patch_bgr[patch_obs] = patch_bg[patch_obs]
        patch_ink[patch_obs] = 0

        if np.sum(patch_ink) < 35:
            continue

        patch_sil = cv2.dilate(patch_ink, np.ones((5, 5), np.uint8), iterations=1)

        other_ink = (obstacle_mask | text_ink_mask).copy()
        other_ink[by1:by2, bx1:bx2][patch_ink > 0] = 0
        other_dilated = cv2.dilate(other_ink, np.ones((5, 5), np.uint8), iterations=1)

        orig_cx = (bx1 + bx2) // 2
        orig_cy = (by1 + by2) // 2

        best_scale = 1.0
        best_comp = 0.0
        best_col_comp = 0.0
        best_dx = 0
        best_dy = 0
        min_col_gap = max(42, int(W * 0.025))

        for s_cand in np.linspace(1.40, 1.05, 15):
            found = False
            for comp_ratio in [0.0, 0.35, 0.70, 1.0]:
                r_bgr, r_sil = compact_white_rows_both(patch_bgr, patch_sil, ratio_remove=comp_ratio, min_keep_run=min_keep)
                for col_ratio in [0.0, 0.30, 0.60]:
                    c_bgr, c_sil = compact_white_cols_both(r_bgr, r_sil, ratio_remove=col_ratio, min_col_gap=min_col_gap)
                    ch, cw = c_bgr.shape[:2]
                    nw = int(round(cw * s_cand))
                    nh = int(round(ch * s_cand))
                    if nw > (x_right_safe - x_left_safe) or nh > (y_bot_safe - y_top_safe):
                        continue
                    scaled_sil = cv2.resize(c_sil, (nw, nh), interpolation=cv2.INTER_NEAREST)

                    for dy in [0, -int(H * 0.006), int(H * 0.006), -int(H * 0.012), int(H * 0.012), -int(H * 0.02), int(H * 0.02), -int(H * 0.03)]:
                        for dx in [0, -int(W * 0.01), int(W * 0.01), -int(W * 0.02), int(W * 0.02), -int(W * 0.035), int(W * 0.035)]:
                            nx1 = orig_cx + dx - nw // 2
                            ny1 = orig_cy + dy - nh // 2
                            if nx1 < x_left_safe or nx1 + nw > x_right_safe or ny1 < y_top_safe or ny1 + nh > y_bot_safe:
                                continue
                            if np.sum(scaled_sil & other_dilated[ny1 : ny1 + nh, nx1 : nx1 + nw]) == 0:
                                best_scale = round(float(s_cand), 2)
                                best_comp = round(float(comp_ratio), 2)
                                best_col_comp = round(float(col_ratio), 2)
                                best_dx = int(dx)
                                best_dy = int(dy)
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break

        detected.append(
            {
                "id": f"block_{idx + 1}",
                "x1": int(bx1),
                "y1": int(by1),
                "x2": int(bx2),
                "y2": int(by2),
                "mode": "image",  # "image" or "font"
                "font_style": "original",
                "scale": best_scale,
                "auto_scale": best_scale,
                "compaction": best_comp,
                "col_compaction": best_col_comp,
                "dx": best_dx,
                "dy": best_dy,
                "boldness": 0.0,
                "darkness": 0.15,
                "sharpness": 0.35,
                "color_override": "",
                # Custom font mode properties
                "text": "",
                "ocr_cells": [],
                "font_family": "Amiri",
                "font_size": 52,
                "font_weight": "700",
                "line_height": 1.65,
                "align": "right",
                "text_color": "#111111",
            }
        )

    return detected


def style_and_scale_patch(
    patch_bgr,
    patch_ink,
    scale=1.0,
    compaction=0.0,
    col_compaction=0.0,
    boldness=0.0,
    darkness=0.0,
    sharpness=0.3,
    color_override="",
    font_style="original",
    min_keep_run=8,
    min_col_gap=45,
):
    """Apply line & column compaction, calligraphic font style, stroke boldness/weight, darkness, color override, and Lanczos4 scaling."""
    c_bgr, c_ink = (
        compact_white_rows_both(patch_bgr, patch_ink, ratio_remove=compaction, min_keep_run=min_keep_run)
        if compaction > 0.01
        else (patch_bgr.copy(), patch_ink.copy())
    )
    if col_compaction > 0.01:
        c_bgr, c_ink = compact_white_cols_both(c_bgr, c_ink, ratio_remove=col_compaction, min_col_gap=min_col_gap)

    out = c_bgr.copy()
    ink_mask_f = cv2.GaussianBlur(c_ink.astype(np.float32), (3, 3), 0)

    # 0. Calligraphic / Font Style reshape (keeps 100% of original harakat & columns while restyling strokes)
    if font_style == "naskh_calligraphic":
        # Horizontal chisel-nib thickening typical of traditional Naskh baseline ribbons
        nib = np.array(
            [
                [0, 0, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 0, 0],
            ],
            dtype=np.uint8,
        )
        chisel = cv2.erode(out, nib, iterations=1)
        out = cv2.addWeighted(out, 0.35, chisel, 0.65, 0)
        ink_mask_f = cv2.dilate(ink_mask_f, nib, iterations=1)
    elif font_style == "ruqah_chisel":
        # Crisp diagonal calligraphy nib
        nib = np.array(
            [
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 0, 0],
            ],
            dtype=np.uint8,
        )
        chisel = cv2.erode(out, nib, iterations=1)
        out = cv2.addWeighted(out, 0.30, chisel, 0.70, 0)
        ink_mask_f = cv2.dilate(ink_mask_f, nib, iterations=1)
    elif font_style == "modern_sans":
        # Uniform monoline rounded stroke feel (like Cairo / Tajawal)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        smoothed = cv2.morphologyEx(out, cv2.MORPH_OPEN, k_open)
        smoothed = cv2.GaussianBlur(smoothed, (3, 3), 0.7)
        out = cv2.addWeighted(out, 0.4, smoothed, 0.6, 0)
        ink_mask_f = cv2.dilate(ink_mask_f, k_open, iterations=1)
    elif font_style == "heavy_textbook":
        # Strong high-contrast textbook print
        k_bold = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(out, k_bold, iterations=1)
        out = cv2.addWeighted(out, 0.25, eroded, 0.75, 0)
        ink_mask_f = cv2.dilate(ink_mask_f, k_bold, iterations=1)

    # 1. Stroke boldness (morphological erosion/dilation on RGB + ink_mask_f so both thickening and thinning work)
    if abs(boldness) >= 0.08:
        k_size = max(2, int(round(abs(boldness) * 1.6)) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        if boldness > 0:
            thickened = cv2.erode(out, kernel, iterations=1)
            alpha = min(1.0, boldness / 2.5)
            out = cv2.addWeighted(out, 1.0 - alpha, thickened, alpha, 0)
            ink_mask_f = cv2.dilate(ink_mask_f, kernel, iterations=1)
        else:
            thinned = cv2.dilate(out, kernel, iterations=1)
            alpha = min(1.0, abs(boldness) / 2.2)
            out = cv2.addWeighted(out, 1.0 - alpha, thinned, alpha, 0)
            eroded_mask = cv2.erode(ink_mask_f, np.ones((2, 2), np.uint8), iterations=1)
            ink_mask_f = ink_mask_f * (1.0 - alpha * 0.5) + eroded_mask * (alpha * 0.5)

    # 2. Ink darkness / contrast boost
    if darkness > 0.01:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_f = l.astype(np.float32) / 255.0
        dark_mask = np.clip((0.82 - l_f) / 0.82, 0.0, 1.0)
        l_new = np.clip(l.astype(np.float32) * (1.0 - darkness * 0.55 * dark_mask), 0, 255).astype(np.uint8)
        out = cv2.cvtColor(cv2.merge([l_new, a, b]), cv2.COLOR_LAB2BGR)

    # Compute continuous ink opacity from local contrast so tinted backgrounds become 0.0 transparent
    bg_local = estimate_local_background(out)
    diff_bgr = np.max(np.abs(out.astype(np.float32) - bg_local.astype(np.float32)), axis=2)
    stroke_alpha = np.clip((diff_bgr - 12.0) / 42.0, 0.0, 1.0)
    stroke_alpha = np.maximum(stroke_alpha, np.clip(ink_mask_f * 0.85, 0.0, 1.0))

    # 3. Optional Ink Color Override (recolor dark letter strokes while preserving colored diacritics if distinct, or recoloring all)
    if color_override and len(color_override) == 7 and color_override.startswith("#"):
        r_val = int(color_override[1:3], 16)
        g_val = int(color_override[3:5], 16)
        b_val = int(color_override[5:7], 16)
        lab_o = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        _, a_o, b_o = cv2.split(lab_o)
        chroma_o = np.sqrt((a_o.astype(np.float32) - 128.0) ** 2 + (b_o.astype(np.float32) - 128.0) ** 2)
        # Keep colored diacritics (e.g. pink/red harakat with chroma > 24) intact while recoloring the dark letters!
        is_colored_haraka = (chroma_o > 24.0)[..., None]
        target_bgr = np.array([[[b_val, g_val, r_val]]], dtype=np.float32)
        recolored = np.clip(bg_local.astype(np.float32) * (1.0 - stroke_alpha[..., None]) + target_bgr * stroke_alpha[..., None], 0, 255).astype(np.uint8)
        out = np.where(is_colored_haraka, out, recolored)

    # 4. Resize by scale
    ch, cw = out.shape[:2]
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    if nw != cw or nh != ch:
        out = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        stroke_alpha = cv2.resize(stroke_alpha, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # 5. Edge sharpness (Unsharp Mask)
    if sharpness > 0.01:
        g = cv2.GaussianBlur(out, (0, 0), sigmaX=0.85)
        w_amount = sharpness * 0.95
        out = cv2.addWeighted(out, 1.0 + w_amount, g, -w_amount, 0)

    return out, np.clip(stroke_alpha, 0.0, 1.0)


def render_blocks_on_page(img_bgr, blocks, margin_trim=0.045):
    """Apply all image-mode blocks (and erase background for font-mode blocks) plus content-safe margin trim."""
    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()
    obstacle_mask, obs_dilated, text_ink_mask, (content_x1, content_y1, content_x2, content_y2) = segment_page_masks(img_bgr)
    bg_bgr = estimate_local_background(img_bgr)
    min_keep = max(6, int(H * 0.004))

    # First pass: extract all patches from original img_bgr and clear original text ink to local paper background color
    prepared = []
    for blk in blocks:
        bx1 = max(0, min(W - 1, int(blk.get("x1", 0))))
        by1 = max(0, min(H - 1, int(blk.get("y1", 0))))
        bx2 = max(bx1 + 1, min(W, int(blk.get("x2", W))))
        by2 = max(by1 + 1, min(H, int(blk.get("y2", H))))

        patch_bgr = img_bgr[by1:by2, bx1:bx2].copy()
        patch_bg = bg_bgr[by1:by2, bx1:bx2]
        patch_ink = text_ink_mask[by1:by2, bx1:bx2].copy()
        patch_obs = obs_dilated[by1:by2, bx1:bx2] > 0

        # If user drew a custom box over an area, exclude obstacles and detect text ink adaptively
        if np.sum(patch_ink) < 20:
            local_white = is_white_mask(patch_bgr)
            patch_ink = ((~local_white) & (~patch_obs)).astype(np.uint8)
        else:
            patch_ink[patch_obs] = 0

        # Replace obstacles inside patch_bgr with clean paper background so obstacles never get duplicated/scaled
        patch_bgr[patch_obs] = patch_bg[patch_obs]

        prepared.append((blk, bx1, by1, bx2, by2, patch_bgr, patch_ink))

        # Clear original text ink inside [by1:by2, bx1:bx2] using the local paper background (never touches obstacles!)
        clear_mask = cv2.dilate(patch_ink, np.ones((7, 7), np.uint8), iterations=1)
        clear_mask[patch_obs] = 0
        clear_alpha = cv2.GaussianBlur(clear_mask.astype(np.float32), (5, 5), 0)[..., None]
        roi_orig = out[by1:by2, bx1:bx2].astype(np.float32)
        out[by1:by2, bx1:bx2] = np.clip(roi_orig * (1.0 - clear_alpha) + patch_bg.astype(np.float32) * clear_alpha, 0, 255).astype(np.uint8)

    # Second pass: composite transformed image blocks
    for (blk, bx1, by1, bx2, by2, patch_bgr, patch_ink) in prepared:
        mode = blk.get("mode", "image")
        if mode != "image":
            # In "font" mode, original text ink is cleanly erased to local paper tint; custom font overlay will be composited
            continue

        scale = float(blk.get("scale", 1.0))
        compaction = float(blk.get("compaction", 0.0))
        col_compaction = float(blk.get("col_compaction", 0.0))
        boldness = float(blk.get("boldness", 0.0))
        darkness = float(blk.get("darkness", 0.15))
        sharpness = float(blk.get("sharpness", 0.35))
        color_override = str(blk.get("color_override", ""))
        font_style = str(blk.get("font_style", "original"))
        dx = int(blk.get("dx", 0))
        dy = int(blk.get("dy", 0))

        styled, styled_alpha = style_and_scale_patch(
            patch_bgr,
            patch_ink,
            scale=scale,
            compaction=compaction,
            col_compaction=col_compaction,
            boldness=boldness,
            darkness=darkness,
            sharpness=sharpness,
            color_override=color_override,
            font_style=font_style,
            min_keep_run=min_keep,
            min_col_gap=max(42, int(W * 0.025)),
        )
        nh, nw = styled.shape[:2]
        blk["render_w"] = int(nw)
        blk["render_h"] = int(nh)
        orig_cx = (bx1 + bx2) // 2
        orig_cy = (by1 + by2) // 2
        nx1 = orig_cx + dx - nw // 2
        ny1 = orig_cy + dy - nh // 2

        # Clamp to page bounds
        nx1 = max(0, min(W - nw, nx1))
        ny1 = max(0, min(H - nh, ny1))
        nh_clip = min(nh, H - ny1)
        nw_clip = min(nw, W - nx1)

        roi = out[ny1 : ny1 + nh_clip, nx1 : nx1 + nw_clip].astype(np.float32)
        s_crop = styled[:nh_clip, :nw_clip].astype(np.float32)
        a_crop = styled_alpha[:nh_clip, :nw_clip, None]
        # Avoid drawing over protected obstacles
        obs_roi = (obstacle_mask[ny1 : ny1 + nh_clip, nx1 : nx1 + nw_clip] > 0)[..., None]
        a_crop = np.where(obs_roi, 0.0, a_crop)
        out[ny1 : ny1 + nh_clip, nx1 : nx1 + nw_clip] = np.clip(roi * (1.0 - a_crop) + s_crop * a_crop, 0, 255).astype(np.uint8)

        content_x1 = min(content_x1, nx1)
        content_y1 = min(content_y1, ny1)
        content_x2 = max(content_x2, nx1 + nw_clip)
        content_y2 = max(content_y2, ny1 + nh_clip)

    # Apply content-safe margin trim if requested
    if margin_trim > 0.002:
        pad_keep_x = int(W * 0.016)
        pad_keep_y = int(H * 0.016)
        max_trim_x = max(0.0, min(content_x1 - pad_keep_x, W - content_x2 - pad_keep_x) / float(W))
        max_trim_y = max(0.0, min(content_y1 - pad_keep_y, H - content_y2 - pad_keep_y) / float(H))
        actual_trim = min(float(margin_trim), max_trim_x, max_trim_y)
        if actual_trim >= 0.005:
            tx = int(round(W * actual_trim))
            ty = int(round(H * actual_trim))
            cropped = out[ty : H - ty, tx : W - tx]
            out = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)

    return out


def ocr_image_patch_detailed(patch_bgr, lang="ar-SA"):
    """Run Windows built-in OCR on an image patch, stripping colored diacritics/obstacles for cleaner word recognition and returning positioned cells."""
    # Pre-clean colored diacritics (pink/red harakat) and background tint so Windows OCR reads Arabic letters cleanly
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    bg_l = cv2.morphologyEx(l, cv2.MORPH_DILATE, k)
    diff_l = bg_l.astype(np.float32) - l.astype(np.float32)
    chroma = np.sqrt((a.astype(np.float32) - 128.0) ** 2 + (b.astype(np.float32) - 128.0) ** 2)

    clean_patch = np.full_like(patch_bgr, 255)
    dark_letters = (diff_l > 24.0) & (chroma < 24.0)
    if np.sum(dark_letters) > 30:
        clean_patch[dark_letters] = patch_bgr[dark_letters]
    else:
        clean_patch = patch_bgr.copy()

    pad = 24
    with tempfile.TemporaryDirectory() as tmpdir:
        padded = cv2.copyMakeBorder(clean_patch, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        img_path = os.path.join(tmpdir, "crop.png")
        out_json = os.path.join(tmpdir, "ocr.json")
        cv2.imwrite(img_path, padded)

        ps_script = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder,Windows.Foundation,ContentType=WindowsRuntime]
$null = [Windows.Globalization.Language,Windows.Foundation,ContentType=WindowsRuntime]

$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {{ $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' }})[0]
Function Await($WinRtTask, $ResultType) {{
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}}

$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync('{img_path}')) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$langObj = New-Object Windows.Globalization.Language('{lang}')
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($langObj)
if ($null -eq $engine) {{
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
}}
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
$items = @()
foreach ($line in $result.Lines) {{
    $minX = 999999; $minY = 999999; $maxX = 0; $maxY = 0
    foreach ($w in $line.Words) {{
        $r = $w.BoundingRect
        if ($r.X -lt $minX) {{ $minX = $r.X }}
        if ($r.Y -lt $minY) {{ $minY = $r.Y }}
        if (($r.X + $r.Width) -gt $maxX) {{ $maxX = $r.X + $r.Width }}
        if (($r.Y + $r.Height) -gt $maxY) {{ $maxY = $r.Y + $r.Height }}
    }}
    $items += @{{
        text = $line.Text
        x = [int]([Math]::Max(0, $minX - {pad}))
        y = [int]([Math]::Max(0, $minY - {pad}))
        w = [int]([Math]::Max(10, $maxX - $minX))
        h = [int]([Math]::Max(10, $maxY - $minY))
    }}
}}
$json = ConvertTo-Json -InputObject @($items) -Depth 4
[System.IO.File]::WriteAllText('{out_json}', $json, [System.Text.Encoding]::UTF8)
"""
        ps_file = os.path.join(tmpdir, "run_ocr.ps1")
        with open(ps_file, "w", encoding="utf-8") as f:
            f.write(ps_script)

        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps_file],
                check=True,
                timeout=15,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if os.path.exists(out_json):
                with open(out_json, "r", encoding="utf-8-sig") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    raw = [raw]
                cells = [c for c in (raw or []) if c and str(c.get("text", "")).strip()]
                if not cells:
                    return {"text": "", "cells": []}

                # Group cells into horizontal rows by vertical center Y
                cells.sort(key=lambda c: c["y"] + c["h"] * 0.5)
                rows = []
                for c in cells:
                    cy = c["y"] + c["h"] * 0.5
                    if not rows:
                        rows.append([c])
                    else:
                        prev_row = rows[-1]
                        prev_cy = sum(x["y"] + x["h"] * 0.5 for x in prev_row) / len(prev_row)
                        avg_h = max(36, sum(x["h"] for x in prev_row) / len(prev_row))
                        if abs(cy - prev_cy) <= avg_h * 0.85:
                            prev_row.append(c)
                        else:
                            rows.append([c])

                is_rtl = lang.lower().startswith("ar")
                ordered_cells = []
                row_strings = []
                for r in rows:
                    # Sort columns RTL (right-to-left) for Arabic, LTR for English
                    r.sort(key=lambda c: c["x"], reverse=is_rtl)
                    ordered_cells.extend(r)
                    row_strings.append("   |   ".join(c["text"].strip() for c in r))

                return {"text": "\n".join(row_strings), "cells": ordered_cells}
        except Exception:
            return {"text": "", "cells": []}
    return {"text": "", "cells": []}


def ocr_image_patch(patch_bgr, lang="ar-SA"):
    return ocr_image_patch_detailed(patch_bgr, lang=lang).get("text", "")
