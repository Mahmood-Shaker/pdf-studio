import os
import cv2
import numpy as np
import subprocess
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REALESRGAN_EXE = os.path.join(BASE_DIR, "bin", "realesrgan", "realesrgan-ncnn-vulkan.exe")


def is_white_mask(img_bgr, l_thresh=235, c_thresh=16):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    chroma = np.sqrt((a.astype(np.float32) - 128.0) ** 2 + (b.astype(np.float32) - 128.0) ** 2)
    return (l >= l_thresh) & (chroma <= c_thresh)


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


def segment_page_masks(img_bgr):
    """Segment page into obstacle_mask, obs_dilated, text_ink_mask, and content bounds."""
    H, W = img_bgr.shape[:2]
    white = is_white_mask(img_bgr, l_thresh=235, c_thresh=16)
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
            (h > int(H * 0.056))
            or (w > int(W * 0.48))
            or (w * h > int(H * W * 0.020))
            or (area > int(H * W * 0.0035))
        )

        if outside_safe or in_page_num_badge or is_illustration:
            obstacle_mask[labels == lbl] = 1
        else:
            text_ink_mask[labels == lbl] = 1

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(W * 0.025), int(H * 0.025)))
    obs_closed = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, close_k)
    obs_dilated = cv2.dilate(obs_closed, np.ones((int(H * 0.014), int(W * 0.014)), np.uint8), iterations=1)
    text_ink_mask[obs_dilated > 0] = 0
    return obs_closed, obs_dilated, text_ink_mask, (content_x1, content_y1, content_x2, content_y2)


def detect_text_blocks(img_bgr):
    """Detect 2D text blocks and calculate each block's recommended auto-fit scale and position."""
    H, W = img_bgr.shape[:2]
    obstacle_mask, obs_dilated, text_ink_mask, _ = segment_page_masks(img_bgr)

    block_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * 0.14), int(H * 0.028)))
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
            if gap_y <= int(H * 0.038) and np.sum(obstacle_mask[yc1 : yc2 + 1, ox1:ox2]) == 0:
                prev[0] = min(prev[0], rb[0])
                prev[1] = min(prev[1], rb[1])
                prev[2] = max(prev[2], rb[2])
                prev[3] = max(prev[3], rb[3])
            else:
                blocks.append(rb)

    pad_src = max(4, int(H * 0.003))
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
        patch_ink = text_ink_mask[by1:by2, bx1:bx2].copy()
        patch_obs = obs_dilated[by1:by2, bx1:bx2] > 0
        patch_bgr[patch_obs] = 255
        patch_ink[patch_obs] = 0

        if np.sum(patch_ink) < 35:
            continue

        sil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * 0.035), safety_px * 2))
        patch_sil = cv2.dilate(patch_ink, sil_kernel, iterations=1)

        other_ink = (obstacle_mask | text_ink_mask).copy()
        other_ink[by1:by2, bx1:bx2][patch_ink > 0] = 0
        other_dilated = cv2.dilate(other_ink, np.ones((safety_px, safety_px), np.uint8), iterations=1)

        orig_cx = (bx1 + bx2) // 2
        orig_cy = (by1 + by2) // 2

        best_scale = 1.0
        best_comp = 0.0
        best_dx = 0
        best_dy = 0

        for s_cand in np.linspace(1.40, 1.05, 15):
            found = False
            for comp_ratio in [0.0, 0.35, 0.70, 1.0]:
                c_bgr, c_sil = compact_white_rows_both(patch_bgr, patch_sil, ratio_remove=comp_ratio, min_keep_run=min_keep)
                ch, cw = c_bgr.shape[:2]
                nw = int(round(cw * s_cand))
                nh = int(round(ch * s_cand))
                if nw > (x_right_safe - x_left_safe) or nh > (y_bot_safe - y_top_safe):
                    continue
                scaled_sil = cv2.resize(c_sil, (nw, nh), interpolation=cv2.INTER_NEAREST)

                for dy in [0, -int(H * 0.01), int(H * 0.01), -int(H * 0.02), int(H * 0.02), -int(H * 0.035), int(H * 0.035), -int(H * 0.05), -int(H * 0.065)]:
                    for dx in [0, -int(W * 0.015), int(W * 0.015), -int(W * 0.03), int(W * 0.03), -int(W * 0.05), int(W * 0.05), -int(W * 0.08), int(W * 0.08), -int(W * 0.12)]:
                        nx1 = orig_cx + dx - nw // 2
                        ny1 = orig_cy + dy - nh // 2
                        if nx1 < x_left_safe or nx1 + nw > x_right_safe or ny1 < y_top_safe or ny1 + nh > y_bot_safe:
                            continue
                        if np.sum(scaled_sil & other_dilated[ny1 : ny1 + nh, nx1 : nx1 + nw]) == 0:
                            best_scale = round(float(s_cand), 2)
                            best_comp = round(float(comp_ratio), 2)
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

        detected.append(
            {
                "id": f"block_{idx + 1}",
                "x1": int(bx1),
                "y1": int(by1),
                "x2": int(bx2),
                "y2": int(by2),
                "mode": "image",  # "image" or "font"
                "scale": best_scale,
                "auto_scale": best_scale,
                "compaction": best_comp,
                "dx": best_dx,
                "dy": best_dy,
                "boldness": 0.0,
                "darkness": 0.15,
                "sharpness": 0.35,
                "color_override": "",
                # Custom font mode properties
                "text": "",
                "font_family": "Amiri",
                "font_size": 36,
                "font_weight": "400",
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
    boldness=0.0,
    darkness=0.0,
    sharpness=0.3,
    color_override="",
    min_keep_run=8,
):
    """Apply line compaction, stroke boldness/weight, darkness/contrast, color override, and Lanczos4 scaling."""
    c_bgr, c_ink = compact_white_rows_both(patch_bgr, patch_ink, ratio_remove=compaction, min_keep_run=min_keep) if (min_keep := min_keep_run) else (patch_bgr, patch_ink)

    out = c_bgr.copy()

    # 1. Stroke boldness (morphological erosion/dilation on RGB so dark strokes thicken or thin smoothly)
    if abs(boldness) >= 0.1:
        k_size = max(2, int(round(abs(boldness) * 1.5)) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        if boldness > 0:
            # Dark strokes have lower pixel values, so ERODE thickens dark ink on white background
            thickened = cv2.erode(out, kernel, iterations=1)
            alpha = min(1.0, boldness / 3.0)
            out = cv2.addWeighted(out, 1.0 - alpha, thickened, alpha, 0)
        else:
            thinned = cv2.dilate(out, kernel, iterations=1)
            alpha = min(1.0, abs(boldness) / 3.0)
            out = cv2.addWeighted(out, 1.0 - alpha, thinned, alpha, 0)

    # 2. Ink darkness / contrast boost
    if darkness > 0.01:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_f = l.astype(np.float32) / 255.0
        dark_mask = np.clip((0.75 - l_f) / 0.75, 0.0, 1.0)
        l_new = np.clip(l.astype(np.float32) * (1.0 - darkness * 0.45 * dark_mask), 0, 255).astype(np.uint8)
        out = cv2.cvtColor(cv2.merge([l_new, a, b]), cv2.COLOR_LAB2BGR)

    # 3. Optional Ink Color Override (recolor non-white strokes to chosen hex color)
    if color_override and len(color_override) == 7 and color_override.startswith("#"):
        r_val = int(color_override[1:3], 16)
        g_val = int(color_override[3:5], 16)
        b_val = int(color_override[5:7], 16)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # Ink alpha: 0 at white (255), 1 at dark ink (0)
        ink_alpha = np.clip((0.92 - gray) / 0.75, 0.0, 1.0)[..., None]
        target_bgr = np.array([[[b_val, g_val, r_val]]], dtype=np.float32)
        white_bgr = np.full_like(out, 255, dtype=np.float32)
        out = np.clip(white_bgr * (1.0 - ink_alpha) + target_bgr * ink_alpha, 0, 255).astype(np.uint8)

    # 4. Resize by scale
    ch, cw = out.shape[:2]
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    if nw != cw or nh != ch:
        out = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    # 5. Edge sharpness (Unsharp Mask)
    if sharpness > 0.01:
        g = cv2.GaussianBlur(out, (0, 0), sigmaX=0.85)
        w_amount = sharpness * 0.85
        out = cv2.addWeighted(out, 1.0 + w_amount, g, -w_amount, 0)

    return out


def render_blocks_on_page(img_bgr, blocks, margin_trim=0.045):
    """Apply all image-mode blocks (and erase background for font-mode blocks) plus content-safe margin trim."""
    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()
    obstacle_mask, obs_dilated, text_ink_mask, (content_x1, content_y1, content_x2, content_y2) = segment_page_masks(img_bgr)
    min_keep = max(6, int(H * 0.004))

    # First pass: extract all patches from original img_bgr and clear original text ink where modified
    prepared = []
    for blk in blocks:
        bx1 = max(0, min(W - 1, int(blk.get("x1", 0))))
        by1 = max(0, min(H - 1, int(blk.get("y1", 0))))
        bx2 = max(bx1 + 1, min(W, int(blk.get("x2", W))))
        by2 = max(by1 + 1, min(H, int(blk.get("y2", H))))

        patch_bgr = img_bgr[by1:by2, bx1:bx2].copy()
        patch_ink = text_ink_mask[by1:by2, bx1:bx2].copy()
        patch_obs = obs_dilated[by1:by2, bx1:bx2] > 0
        # If user drew a custom box over something not in text_ink_mask, compute local ink mask
        if np.sum(patch_ink) < 20:
            local_white = is_white_mask(patch_bgr, l_thresh=235, c_thresh=16)
            patch_ink = (~local_white).astype(np.uint8)
        else:
            patch_bgr[patch_obs] = 255
            patch_ink[patch_obs] = 0

        prepared.append((blk, bx1, by1, bx2, by2, patch_bgr, patch_ink))

        # Clear original text ink inside [by1:by2, bx1:bx2]
        clear_mask = cv2.dilate(patch_ink, np.ones((5, 5), np.uint8), iterations=1)
        out[by1:by2, bx1:bx2][clear_mask > 0] = 255

    # Second pass: composite transformed image blocks
    for (blk, bx1, by1, bx2, by2, patch_bgr, patch_ink) in prepared:
        mode = blk.get("mode", "image")
        if mode != "image":
            # In "font" mode, the original ink is now erased to clean white; custom font overlay will be composited
            continue

        scale = float(blk.get("scale", 1.0))
        compaction = float(blk.get("compaction", 0.0))
        boldness = float(blk.get("boldness", 0.0))
        darkness = float(blk.get("darkness", 0.15))
        sharpness = float(blk.get("sharpness", 0.35))
        color_override = str(blk.get("color_override", ""))
        dx = int(blk.get("dx", 0))
        dy = int(blk.get("dy", 0))

        styled = style_and_scale_patch(
            patch_bgr,
            patch_ink,
            scale=scale,
            compaction=compaction,
            boldness=boldness,
            darkness=darkness,
            sharpness=sharpness,
            color_override=color_override,
            min_keep_run=min_keep,
        )
        nh, nw = styled.shape[:2]
        orig_cx = (bx1 + bx2) // 2
        orig_cy = (by1 + by2) // 2
        nx1 = orig_cx + dx - nw // 2
        ny1 = orig_cy + dy - nh // 2

        # Clamp to page bounds
        nx1 = max(0, min(W - nw, nx1))
        ny1 = max(0, min(H - nh, ny1))
        nh_clip = min(nh, H - ny1)
        nw_clip = min(nw, W - nx1)

        roi = out[ny1 : ny1 + nh_clip, nx1 : nx1 + nw_clip]
        out[ny1 : ny1 + nh_clip, nx1 : nx1 + nw_clip] = np.minimum(roi, styled[:nh_clip, :nw_clip])

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


def ocr_image_patch(patch_bgr, lang="ar-SA"):
    """Run Windows built-in OCR (ar-SA or en-US) on an image patch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Pad with white border for cleaner OCR recognition
        padded = cv2.copyMakeBorder(patch_bgr, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        img_path = os.path.join(tmpdir, "crop.png")
        out_txt = os.path.join(tmpdir, "ocr.txt")
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
$lines = @()
foreach ($line in $result.Lines) {{
    $words = @()
    foreach ($w in $line.Words) {{
        $words += $w.Text
    }}
    if ('{lang}' -like 'ar*') {{
        [array]::Reverse($words)
    }}
    $lines += ($words -join ' ')
}}
[System.IO.File]::WriteAllLines('{out_txt}', $lines, [System.Text.Encoding]::UTF8)
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
            if os.path.exists(out_txt):
                with open(out_txt, "r", encoding="utf-8-sig") as f:
                    return f.read().strip()
        except Exception as e:
            return f""
    return ""
