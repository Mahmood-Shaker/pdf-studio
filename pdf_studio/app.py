import os
import io
import base64
import time
import threading
import pymupdf
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Body, HTTPException
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

import engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
STATIC_DIR = os.path.join(BASE_DIR, "static")
FONTS_DIR = os.path.join(BASE_DIR, "fonts")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(FONTS_DIR, exist_ok=True)

app = FastAPI(title="PDF Text, Font & Resolution Studio")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/fonts", StaticFiles(directory=FONTS_DIR), name="fonts")

# Active PDF session state
STATE: Dict[str, Any] = {
    "pdf_name": None,
    "pdf_path": None,
    "page_count": 0,
    # Cache of committed page BGR images (key: page_index 0-based)
    "committed_pages": {},
    # Cache of detected/edited blocks per page
    "page_blocks": {},
    # Batch job status
    "batch": {
        "running": False,
        "current": 0,
        "total": 0,
        "status": "idle",
        "output_file": "",
        "error": "",
    },
}


def get_page_base_bgr(page_idx: int, use_committed: bool = True) -> np.ndarray:
    """Render page at 3x matrix (~216 DPI) so both 72-DPI and 300-DPI PDFs are crisp."""
    if use_committed and page_idx in STATE["committed_pages"]:
        return STATE["committed_pages"][page_idx].copy()

    if not STATE["pdf_path"] or not os.path.exists(STATE["pdf_path"]):
        raise HTTPException(status_code=400, detail="No PDF file is currently open.")

    doc = pymupdf.open(STATE["pdf_path"])
    if page_idx < 0 or page_idx >= len(doc):
        doc.close()
        raise HTTPException(status_code=404, detail="Page index out of range.")

    page = doc[page_idx]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(3, 3))
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    doc.close()
    return img_bgr


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/files")
def list_files():
    pdfs = []
    for fname in sorted(os.listdir(WORKSPACE_DIR)):
        if fname.lower().endswith(".pdf"):
            fpath = os.path.join(WORKSPACE_DIR, fname)
            if os.path.isfile(fpath):
                size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
                pdfs.append({"name": fname, "size_mb": size_mb})

    custom_fonts = []
    for fname in sorted(os.listdir(FONTS_DIR)):
        if fname.lower().endswith((".ttf", ".otf", ".woff", ".woff2")):
            font_family = os.path.splitext(fname)[0]
            custom_fonts.append({"name": font_family, "file": fname, "url": f"/fonts/{fname}"})

    return {
        "pdfs": pdfs,
        "custom_fonts": custom_fonts,
        "active_pdf": STATE["pdf_name"],
        "page_count": STATE["page_count"],
    }


@app.post("/api/open_pdf")
def open_pdf(payload: Dict[str, Any] = Body(...)):
    filename = payload.get("filename", "")
    fpath = os.path.join(WORKSPACE_DIR, filename)
    if not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail=f"File {filename} not found.")

    doc = pymupdf.open(fpath)
    page_count = len(doc)
    doc.close()

    STATE["pdf_name"] = filename
    STATE["pdf_path"] = fpath
    STATE["page_count"] = page_count
    STATE["committed_pages"] = {}
    STATE["page_blocks"] = {}

    return {"status": "ok", "filename": filename, "page_count": page_count}


@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    fname = os.path.basename(file.filename or "uploaded.pdf")
    if not fname.lower().endswith(".pdf"):
        fname += ".pdf"
    dest = os.path.join(WORKSPACE_DIR, fname)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    return open_pdf({"filename": fname})


@app.post("/api/upload_font")
async def upload_font(file: UploadFile = File(...)):
    fname = os.path.basename(file.filename or "custom.ttf")
    if not fname.lower().endswith((".ttf", ".otf", ".woff", ".woff2")):
        raise HTTPException(status_code=400, detail="Only .ttf, .otf, .woff, .woff2 font files are supported.")
    dest = os.path.join(FONTS_DIR, fname)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    font_family = os.path.splitext(fname)[0]
    return {"status": "ok", "font": {"name": font_family, "file": fname, "url": f"/fonts/{fname}"}}


@app.get("/api/page/{page_num}/image")
def get_page_image(page_num: int, original: int = 0):
    page_idx = page_num - 1
    img_bgr = get_page_base_bgr(page_idx, use_committed=(original == 0))
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/page/{page_num}/detect_blocks")
def detect_page_blocks(page_num: int, force: int = 0):
    page_idx = page_num - 1
    img_bgr = get_page_base_bgr(page_idx, use_committed=False)
    H, W = img_bgr.shape[:2]

    if force == 0 and page_idx in STATE["page_blocks"]:
        blocks = STATE["page_blocks"][page_idx]
    else:
        blocks = engine.detect_text_blocks(img_bgr)
        STATE["page_blocks"][page_idx] = blocks

    return {
        "page_num": page_num,
        "width": W,
        "height": H,
        "blocks": blocks,
        "is_committed": page_idx in STATE["committed_pages"],
    }


@app.post("/api/page/{page_num}/ocr")
def ocr_block(page_num: int, payload: Dict[str, Any] = Body(...)):
    page_idx = page_num - 1
    img_bgr = get_page_base_bgr(page_idx, use_committed=False)
    H, W = img_bgr.shape[:2]

    x1 = max(0, min(W - 1, int(payload.get("x1", 0))))
    y1 = max(0, min(H - 1, int(payload.get("y1", 0))))
    x2 = max(x1 + 1, min(W, int(payload.get("x2", W))))
    y2 = max(y1 + 1, min(H, int(payload.get("y2", H))))
    lang = payload.get("lang", "ar-SA")

    # First check if native text exists in PDF at that rect
    doc = pymupdf.open(STATE["pdf_path"])
    page = doc[page_idx]
    rect = pymupdf.Rect(x1 / 3.0, y1 / 3.0, x2 / 3.0, y2 / 3.0)
    native_text = page.get_text("text", clip=rect).strip()
    doc.close()

    if native_text:
        return {"text": native_text, "cells": [], "source": "pdf_text"}

    patch_bgr = img_bgr[y1:y2, x1:x2]
    res = engine.ocr_image_patch_detailed(patch_bgr, lang=lang)
    return {"text": res.get("text", ""), "cells": res.get("cells", []), "source": "windows_ocr"}


@app.post("/api/page/{page_num}/render")
def render_page_edits(page_num: int, payload: Dict[str, Any] = Body(...)):
    """
    Render the page with all current blocks (both 'image' mode and 'font' overlay mode),
    optional margin_trim, and optionally commit to STATE['committed_pages'].
    """
    page_idx = page_num - 1
    base_bgr = get_page_base_bgr(page_idx, use_committed=False)
    H, W = base_bgr.shape[:2]

    blocks = payload.get("blocks", [])
    margin_trim = float(payload.get("margin_trim", 0.045))
    commit = bool(payload.get("commit", False))
    font_overlay_b64 = payload.get("font_overlay_png", "")

    STATE["page_blocks"][page_idx] = blocks

    # Render image-mode blocks and erase original ink under font-mode blocks
    # First render without margin_trim if font_overlay needs to be composited in page coordinates
    rendered_bgr = engine.render_blocks_on_page(base_bgr, blocks, margin_trim=0.0)

    # Composite high-DPI custom font transparent PNG overlay generated by browser canvas
    if font_overlay_b64 and "," in font_overlay_b64:
        raw_b64 = font_overlay_b64.split(",", 1)[1]
        png_bytes = base64.b64decode(raw_b64)
        overlay_rgba = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if overlay_rgba is not None and overlay_rgba.shape[2] == 4:
            if overlay_rgba.shape[:2] != (H, W):
                overlay_rgba = cv2.resize(overlay_rgba, (W, H), interpolation=cv2.INTER_LANCZOS4)
            alpha = (overlay_rgba[:, :, 3:4].astype(np.float32)) / 255.0
            overlay_bgr = overlay_rgba[:, :, :3].astype(np.float32)
            rendered_bgr = np.clip(rendered_bgr.astype(np.float32) * (1.0 - alpha) + overlay_bgr * alpha, 0, 255).astype(np.uint8)

    # Now apply uniform margin_trim if requested
    if margin_trim > 0.002:
        tx = int(round(W * min(0.08, margin_trim)))
        ty = int(round(H * min(0.08, margin_trim)))
        if tx > 2 and ty > 2:
            cropped = rendered_bgr[ty : H - ty, tx : W - tx]
            rendered_bgr = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LANCZOS4)

    if commit:
        STATE["committed_pages"][page_idx] = rendered_bgr

    ok, buf = cv2.imencode(".jpg", rendered_bgr, [cv2.IMWRITE_JPEG_QUALITY, 93])
    b64_jpg = base64.b64encode(buf.tobytes()).decode("ascii")
    return {
        "status": "ok",
        "committed": commit,
        "blocks": blocks,
        "image_data_url": f"data:image/jpeg;base64,{b64_jpg}",
    }


@app.post("/api/page/{page_num}/reset")
def reset_page(page_num: int):
    page_idx = page_num - 1
    STATE["committed_pages"].pop(page_idx, None)
    STATE["page_blocks"].pop(page_idx, None)
    return {"status": "ok"}


@app.post("/api/page/{page_num}/enhance_sr")
def enhance_single_page_sr(page_num: int):
    """Run AI Super-Resolution + Background Cleaning on the current page."""
    page_idx = page_num - 1
    if not STATE["pdf_path"]:
        raise HTTPException(status_code=400, detail="No PDF open.")

    doc = pymupdf.open(STATE["pdf_path"])
    page = doc[page_idx]
    pix = page.get_pixmap()  # 1x native resolution
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    doc.close()

    enhanced_bgr = engine.enhance_page_resolution(img_bgr, scale=3, use_gpu=True)
    STATE["committed_pages"][page_idx] = enhanced_bgr
    ok, buf = cv2.imencode(".jpg", enhanced_bgr, [cv2.IMWRITE_JPEG_QUALITY, 93])
    b64_jpg = base64.b64encode(buf.tobytes()).decode("ascii")
    return {"status": "ok", "image_data_url": f"data:image/jpeg;base64,{b64_jpg}"}


@app.post("/api/save_pdf")
def save_edited_pdf(payload: Dict[str, Any] = Body(...)):
    """Save the open PDF with all committed page edits to a file in the workspace."""
    if not STATE["pdf_path"]:
        raise HTTPException(status_code=400, detail="No PDF open.")

    out_name = payload.get("output_filename", "").strip()
    if not out_name:
        base, _ = os.path.splitext(STATE["pdf_name"])
        out_name = f"{base}_edited.pdf"
    if not out_name.lower().endswith(".pdf"):
        out_name += ".pdf"

    out_path = os.path.join(WORKSPACE_DIR, out_name)
    doc = pymupdf.open(STATE["pdf_path"])

    for page_idx, bgr in STATE["committed_pages"].items():
        if 0 <= page_idx < len(doc):
            page = doc[page_idx]
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            img_bytes = buf.tobytes()
            imgs = page.get_images()
            if imgs:
                page.replace_image(imgs[0][0], stream=img_bytes)
            else:
                page.clean_contents()
                page.insert_image(page.rect, stream=img_bytes)

    # Save safely even if overwriting same path
    tmp_path = out_path + ".tmp"
    doc.save(tmp_path, garbage=4, deflate=True)
    doc.close()
    os.replace(tmp_path, out_path)

    size_mb = round(os.path.getsize(out_path) / (1024 * 1024), 2)
    return {
        "status": "ok",
        "filename": out_name,
        "size_mb": size_mb,
        "download_url": f"/api/download/{out_name}",
    }


@app.get("/api/download/{filename}")
def download_pdf(filename: str):
    fpath = os.path.join(WORKSPACE_DIR, os.path.basename(filename))
    if not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(fpath, media_type="application/pdf", filename=os.path.basename(filename))


@app.post("/api/batch_start")
def start_batch_job(payload: Dict[str, Any] = Body(...)):
    if STATE["batch"]["running"]:
        raise HTTPException(status_code=400, detail="A batch job is already running.")
    if not STATE["pdf_path"]:
        raise HTTPException(status_code=400, detail="No PDF open.")

    start_page = max(1, int(payload.get("start_page", 1)))
    end_page = min(STATE["page_count"], int(payload.get("end_page", STATE["page_count"])))
    max_scale = float(payload.get("max_scale", 1.35))
    boldness = float(payload.get("boldness", 0.0))
    darkness = float(payload.get("darkness", 0.15))
    sharpness = float(payload.get("sharpness", 0.35))
    margin_trim = float(payload.get("margin_trim", 0.045))
    color_override = str(payload.get("color_override", ""))
    clean_bg = bool(payload.get("clean_bg", False))
    out_name = str(payload.get("output_filename", "")).strip()
    if not out_name:
        base, _ = os.path.splitext(STATE["pdf_name"])
        out_name = f"{base}_custom_batch.pdf"
    if not out_name.lower().endswith(".pdf"):
        out_name += ".pdf"

    STATE["batch"] = {
        "running": True,
        "current": 0,
        "total": end_page - start_page + 1,
        "status": "Starting batch processing...",
        "output_file": out_name,
        "error": "",
    }

    def worker():
        try:
            doc = pymupdf.open(STATE["pdf_path"])
            for idx, pno in enumerate(range(start_page, end_page + 1)):
                page_idx = pno - 1
                STATE["batch"]["current"] = idx + 1
                STATE["batch"]["status"] = f"Processing page {pno} of {end_page}..."

                page = doc[page_idx]
                pix = page.get_pixmap(matrix=pymupdf.Matrix(3, 3))
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                img_bgr = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2BGR)

                if clean_bg:
                    img_bgr = engine.pre_clean_page(img_bgr)

                if page_idx > 0:
                    blocks = engine.detect_text_blocks(img_bgr)
                    for b in blocks:
                        b["scale"] = min(float(b["auto_scale"]), max_scale)
                        b["boldness"] = boldness
                        b["darkness"] = darkness
                        b["sharpness"] = sharpness
                        b["color_override"] = color_override
                    out_bgr = engine.render_blocks_on_page(img_bgr, blocks, margin_trim=margin_trim)
                else:
                    out_bgr = img_bgr

                ok, buf = cv2.imencode(".jpg", out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                img_bytes = buf.tobytes()
                imgs = page.get_images()
                if imgs:
                    page.replace_image(imgs[0][0], stream=img_bytes)
                else:
                    page.clean_contents()
                    page.insert_image(page.rect, stream=img_bytes)

            out_path = os.path.join(WORKSPACE_DIR, out_name)
            tmp_path = out_path + ".tmp"
            STATE["batch"]["status"] = "Saving output PDF..."
            doc.save(tmp_path, garbage=4, deflate=True)
            doc.close()
            os.replace(tmp_path, out_path)
            STATE["batch"]["status"] = "completed"
            STATE["batch"]["running"] = False
        except Exception as e:
            STATE["batch"]["error"] = str(e)
            STATE["batch"]["status"] = "error"
            STATE["batch"]["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return {"status": "started", "output_file": out_name}


@app.get("/api/batch_status")
def get_batch_status():
    return STATE["batch"]
