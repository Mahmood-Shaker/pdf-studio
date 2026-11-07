import io
import cv2
import numpy as np
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

import engine


def set_paragraph_rtl(paragraph, align=WD_ALIGN_PARAGRAPH.RIGHT):
    """Enable native OpenXML Right-to-Left (<w:bidi/>) on a Word paragraph."""
    paragraph.alignment = align
    pPr = paragraph._p.get_or_add_pPr()
    bidi = pPr.find(qn("w:bidi"))
    if bidi is None:
        bidi = OxmlElement("w:bidi")
        pPr.append(bidi)


def add_rtl_run(paragraph, text, font_name="Traditional Arabic", font_size_pt=20, bold=True, color_hex="#111111"):
    """Add an Arabic RTL text run with explicit Complex Script (w:cs / w:szCs / w:rtl) OpenXML properties."""
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = font_name
    run.font.size = Pt(font_size_pt)

    if color_hex and len(color_hex) == 7 and color_hex.startswith("#"):
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        run.font.color.rgb = RGBColor(r, g, b)

    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), font_name)
    rFonts.set(qn("w:hAnsi"), font_name)
    rFonts.set(qn("w:cs"), font_name)

    sz_val = str(int(font_size_pt * 2))
    for tag in ("w:sz", "w:szCs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = OxmlElement(tag)
            rPr.append(el)
        el.set(qn("w:val"), sz_val)

    if bold:
        bCs = rPr.find(qn("w:bCs"))
        if bCs is None:
            bCs = OxmlElement("w:bCs")
            rPr.append(bCs)

    rtl = rPr.find(qn("w:rtl"))
    if rtl is None:
        rtl = OxmlElement("w:rtl")
        rPr.append(rtl)

    return run


def set_table_rtl_and_borders(table, show_grid=True):
    """Set OpenXML <w:bidiVisual/> on a Word table so column 0 is on the RIGHT (RTL flow)."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = table._tbl.tblPr
    if tblPr is None:
        tblPr = OxmlElement("w:tblPr")
        table._tbl.insert(0, tblPr)

    bidiVisual = tblPr.find(qn("w:bidiVisual"))
    if bidiVisual is None:
        bidiVisual = OxmlElement("w:bidiVisual")
        tblPr.append(bidiVisual)

    if show_grid:
        tblBorders = tblPr.find(qn("w:tblBorders"))
        if tblBorders is None:
            tblBorders = OxmlElement("w:tblBorders")
            tblPr.append(tblBorders)
        for border_name in ("top", "left", "bottom", "right", "insideH", "insideV"):
            border = OxmlElement(f"w:{border_name}")
            border.set(qn("w:val"), "single")
            border.set(qn("w:sz"), "4")
            border.set(qn("w:space"), "0")
            border.set(qn("w:color"), "CBD5E1")
            tblBorders.append(border)


def detect_interior_graphic_boxes(img_bgr, text_blocks):
    """Detect interior colored headers (e.g., pink 'إقرأ واكتب :' badges) and illustrations inside the page frame."""
    H, W = img_bgr.shape[:2]
    white = engine.is_white_mask(img_bgr)
    ink = (~white).astype(np.uint8)

    # Strip long vertical and horizontal outer frame borders so top headers touching the frame border are not discarded
    vert_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((int(H * 0.25), 1), np.uint8))
    horiz_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, int(W * 0.35)), np.uint8))
    border_mask = cv2.dilate(vert_lines | horiz_lines, np.ones((15, 15), np.uint8))

    # Ignore outer page margin (outer 6% of page width, top 4%, bottom 9% page-number footer)
    mx = int(W * 0.06)
    my_top = int(H * 0.04)
    my_bot = int(H * 0.91)
    interior = np.zeros_like(ink)
    interior[my_top:my_bot, mx : W - mx] = ink[my_top:my_bot, mx : W - mx]
    interior[border_mask > 0] = 0

    # Exclude regions already covered by detected text blocks
    for b in text_blocks:
        bx1 = max(0, int(b["x1"]) - 6)
        by1 = max(0, int(b["y1"]) - 6)
        bx2 = min(W, int(b["x2"]) + 6)
        by2 = min(H, int(b["y2"]) + 6)
        interior[by1:by2, bx1:bx2] = 0

    # Also remove thin horizontal dotted separator rules
    horiz_close = cv2.morphologyEx(interior, cv2.MORPH_CLOSE, np.ones((1, 25), np.uint8))
    nh, lh, sth, _ = cv2.connectedComponentsWithStats(horiz_close, connectivity=8)
    for i in range(1, nh):
        hx, hy, hw, hh, _ = sth[i]
        if hw > int(W * 0.35) and hh < int(H * 0.015):
            interior[lh == i] = 0

    closed = cv2.morphologyEx(interior, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    nl, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    graphics = []
    for i in range(1, nl):
        x, y, w, h, area = stats[i]
        # Skip outer border fragments that span >65% of page height or >85% of page width
        if h > int(H * 0.65) or w > int(W * 0.85):
            continue
        if area >= int(H * W * 0.0020) and w >= int(W * 0.05) and h >= int(H * 0.02):
            pad = 6
            gx1 = max(0, x - pad)
            gy1 = max(0, y - pad)
            gx2 = min(W, x + w + pad)
            gy2 = min(H, y + h + pad)
            graphics.append({"type": "graphic", "x1": gx1, "y1": gy1, "x2": gx2, "y2": gy2})

    return graphics


def add_cv2_image_to_doc(doc, patch_bgr, width_inches):
    ok, buf = cv2.imencode(".png", patch_bgr)
    if not ok:
        return
    stream = io.BytesIO(buf.tobytes())
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(stream, width=Inches(width_inches))


def parse_block_to_rows_table(img_bgr, blk):
    """Extract 2D RTL table rows/columns for a block (from user's custom text if present, otherwise via RTL OCR)."""
    user_txt = str(blk.get("text", "")).strip()
    if user_txt:
        rows_table = []
        for line in user_txt.split("\n"):
            if "|" in line:
                cols = [c.strip() for c in line.split("|") if c.strip()]
                if cols:
                    rows_table.append(cols)
            elif line.strip():
                rows_table.append([line.strip()])
        if rows_table:
            return rows_table

    H, W = img_bgr.shape[:2]
    x1 = max(0, min(W - 1, int(blk.get("x1", 0))))
    y1 = max(0, min(H - 1, int(blk.get("y1", 0))))
    x2 = max(x1 + 1, min(W, int(blk.get("x2", W))))
    y2 = max(y1 + 1, min(H, int(blk.get("y2", H))))

    res = engine.ocr_image_patch_detailed(img_bgr[y1:y2, x1:x2], lang="ar-SA")
    return res.get("rows_table", [])


def append_page_to_docx(
    doc: Document,
    orig_bgr: np.ndarray,
    rendered_bgr: np.ndarray,
    blocks: list,
    mode: str = "editable",
    font_name: str = "Traditional Arabic",
    font_size_pt: int = 20,
    is_first_page: bool = True,
):
    """Append a single PDF page to the Word Document: OCR text tables for text blocks + cropped images for non-text graphics."""
    if not is_first_page:
        doc.add_page_break()

    H, W = orig_bgr.shape[:2]

    if mode == "visual":
        add_cv2_image_to_doc(doc, rendered_bgr, width_inches=6.5)
        return

    graphics = detect_interior_graphic_boxes(orig_bgr, blocks)
    elements = list(graphics)
    for b in blocks:
        elements.append({"type": "block", "y1": int(b.get("y1", 0)), "block": b})

    # Sort all visual headers/illustrations and text blocks top-to-bottom
    elements.sort(key=lambda el: el["y1"])

    for el in elements:
        if el["type"] == "graphic":
            gx1, gy1, gx2, gy2 = el["x1"], el["y1"], el["x2"], el["y2"]
            crop = orig_bgr[gy1:gy2, gx1:gx2]
            w_in = max(0.95, min(3.2, ((gx2 - gx1) / float(W)) * 4.8))
            add_cv2_image_to_doc(doc, crop, width_inches=w_in)
        else:
            b = el["block"]
            bx1 = max(0, min(W - 1, int(b.get("x1", 0))))
            by1 = max(0, min(H - 1, int(b.get("y1", 0))))
            bx2 = max(bx1 + 1, min(W, int(b.get("x2", W))))
            by2 = max(by1 + 1, min(H, int(b.get("y2", H))))

            if mode == "hybrid_crops":
                crop_blk = rendered_bgr[by1:by2, bx1:bx2]
                w_in = max(2.2, min(5.5, ((bx2 - bx1) / float(W)) * 5.8))
                add_cv2_image_to_doc(doc, crop_blk, width_inches=w_in)

            rows_table = parse_block_to_rows_table(orig_bgr, b)
            if not rows_table:
                continue

            max_cols = max(len(r) for r in rows_table)
            base_color = str(b.get("color_override") or b.get("text_color") or "#111111")

            if max_cols == 1:
                for r in rows_table:
                    p = doc.add_paragraph()
                    set_paragraph_rtl(p, align=WD_ALIGN_PARAGRAPH.RIGHT)
                    p.paragraph_format.space_before = Pt(1)
                    p.paragraph_format.space_after = Pt(2)
                    add_rtl_run(p, r[0], font_name=font_name, font_size_pt=font_size_pt, bold=True, color_hex=base_color)
            else:
                table = doc.add_table(rows=len(rows_table), cols=max_cols)
                set_table_rtl_and_borders(table, show_grid=True)
                for r_idx, r_cols in enumerate(rows_table):
                    row_cells = table.rows[r_idx].cells
                    for c_idx in range(max_cols):
                        cell = row_cells[c_idx]
                        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                        p = cell.paragraphs[0]
                        set_paragraph_rtl(p, align=WD_ALIGN_PARAGRAPH.CENTER)
                        p.paragraph_format.space_before = Pt(1)
                        p.paragraph_format.space_after = Pt(1)
                        txt = r_cols[c_idx] if c_idx < len(r_cols) else ""
                        if txt:
                            cell_color = "#C026D3" if (txt.endswith("ـ") and len(txt) <= 4) else base_color
                            add_rtl_run(p, txt, font_name=font_name, font_size_pt=font_size_pt, bold=True, color_hex=cell_color)


def create_rtl_document() -> Document:
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.35)
        section.bottom_margin = Inches(0.35)
        section.left_margin = Inches(0.5)
        section.right_margin = Inches(0.5)
    return doc
