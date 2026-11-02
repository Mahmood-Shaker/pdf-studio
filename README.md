# PDF Text, Font & Resolution Studio

An interactive web studio for enhancing scanned PDFs, enlarging and reflowing text blocks while preserving colored callouts/tables, editing text inline with custom Arabic & Latin fonts, and upscaling page resolution (OpenCV / Real-ESRGAN).

## Features
- **Smart Text Detection & Reflow**: Automatically isolates standard dark text lines from colored headers, badges, and tables, enlarging and reflowing text cleanly within safe page margins.
- **Interactive Block Editor**: Move, resize, scale, bold, recolor, delete, or replace any text block with live rich-text editing and custom `.ttf`/`.otf` fonts.
- **Resolution & Quality Enhancement**: Page de-noising, contrast enhancement, sharpening, and AI super-resolution support (Real-ESRGAN Vulkan or OpenCV Lanczos4).
- **Batch Processing & Export**: Apply enhancements across any page range and export high-DPI PDFs.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the studio:
   - **Windows**: Double-click `run_pdf_studio.bat`
   - **CLI**:
     ```bash
     python -m uvicorn app:app --app-dir pdf_studio --host 127.0.0.1 --port 8765
     ```
3. Open `http://127.0.0.1:8765` in your browser.
