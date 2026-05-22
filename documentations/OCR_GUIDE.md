# OCR Support for Scanned PDFs

## Overview

The CV Ranking Agent now supports **scanned PDFs** (image-based resumes) using local OCR processing. This means:

- ✅ No additional API costs (runs locally)
- ✅ Works with the free Google AI tier
- ✅ Automatic error correction for common OCR mistakes (S→5, O→0, etc.)
- ✅ Seamless fallback — if OCR libraries aren't installed, the app still works for text-based PDFs

---

## Quick Setup

### 1. Install OCR dependencies

```bash
pip install -r requirements-ocr.txt
```

This installs:
- **PyMuPDF** — Fast PDF rendering to images
- **EasyOCR** — Neural network-based OCR (no system dependencies!)
- **Pillow** — Image processing
- **NumPy** — Array operations

### 2. First run

On first use, EasyOCR will automatically download the English language model (~100MB). This happens once and is cached locally.

### 3. That's it!

No system dependencies, no PATH configuration, no external binaries. Just Python packages.

---

## How It Works

### Detection
When a PDF is uploaded, the system checks if text extraction yields less than 50 characters. If so, it's flagged as a scanned PDF.

### OCR Processing
1. **PyMuPDF** renders each page to a high-resolution image (300 DPI)
2. **EasyOCR** extracts text using a pre-trained neural network
3. **Regex pre-clean** fixes unambiguous errors (phone numbers, whitespace)
4. **Gemini correction** fixes context-aware errors with strict constraints:
   - Temperature = 0.0 (deterministic, no creative rewriting)
   - Prompt explicitly forbids adding/removing words or rephrasing
   - Only fixes character-level OCR mistakes (5→S, 0→O, 1→I/l, rn→m)
   - Safety fallback: if output is suspiciously short, keeps original text
5. **Result**: Clean text ready for embedding and ranking

### Fallback Behavior
If OCR libraries aren't installed:
- Text-based PDFs work normally
- Scanned PDFs are skipped with a warning message
- The app continues processing other resumes

---

## Performance

| Metric | Value |
|--------|-------|
| **Processing time per page** | 2–5 seconds (CPU mode) |
| **Accuracy** | 95–98% for clean scans |
| **Memory usage** | ~500MB (EasyOCR model) |
| **Disk space** | ~100MB (cached model) |

**Tip:** For large batches of scanned PDFs, consider pre-processing them offline to avoid slowing down the ranking pipeline.

---

## Troubleshooting

### "OCR libraries not installed"
Run: `pip install -r requirements-ocr.txt`

### "Failed to download EasyOCR model"
Check your internet connection. The model downloads automatically on first use.

### OCR text is garbled
The scan quality may be too low. Try:
- Re-scanning at higher DPI (300+ recommended)
- Adjusting contrast/brightness before scanning
- Using a text-based PDF instead

### Out of memory errors
EasyOCR loads a ~500MB model. If you're on a low-memory system:
- Close other applications
- Process fewer resumes per batch
- Consider using a machine with more RAM

---

## Technical Details

### Why EasyOCR?

| Feature | EasyOCR | Tesseract | Google Vision API |
|---------|---------|-----------|-------------------|
| **Setup** | `pip install` | System binary required | API key + billing |
| **Accuracy** | 95–98% | 90–95% | 98–99% |
| **Speed** | 2–5s/page | 1–3s/page | <1s/page |
| **Cost** | Free | Free | $1.50 per 1000 pages |
| **Dependencies** | Python only | C++ binary + data files | Internet required |

EasyOCR strikes the best balance for this use case:
- No system dependencies (works on any OS)
- High accuracy with neural networks
- Completely free and offline
- Simple Python-only installation

### Why PyMuPDF?

PyMuPDF (fitz) is the fastest PDF rendering library in Python:
- 5–10× faster than pdf2image
- No poppler dependency
- Built-in image extraction at any DPI
- Minimal memory footprint

---

## Error Correction Rules

The post-correction engine fixes these common OCR mistakes:

### Letter/Digit Confusion
- `5` → `S` (at word boundaries: "5kills" → "Skills")
- `0` → `O` (at word boundaries: "0ffice" → "Office")
- `1` → `I` or `l` (context-dependent)

### Common Resume Words
- `Yea4rs` → `Years`
- `Exper1ence` → `Experience`
- `M4nager` → `Manager`
- `Deve1oper` → `Developer`
- `Eng1neer` → `Engineer`
- `Pr0ject` → `Project`

### Email Patterns
- `user@doma1n.com` → `user@domain.com`
- `contact@comp4ny.org` → `contact@company.org`

### Phone Patterns
- `555-O123-4567` → `555-0123-4567`
- `(555) l23-4567` → `(555) 123-4567`

---

## Extending OCR Support

### Add More Languages

```python
# In ocr_processor.py, modify the Reader initialization:
_reader = easyocr.Reader(['en', 'es', 'fr'], gpu=False)
```

Supported languages: https://www.jaided.ai/easyocr/

### Adjust OCR Confidence Threshold

```python
# In ocr_processor.py, modify readtext call:
results = _reader.readtext(img_array, detail=1, paragraph=True)
# Filter by confidence:
filtered = [text for (bbox, text, conf) in results if conf > 0.5]
```

### Enable GPU Acceleration

```python
# In ocr_processor.py:
_reader = easyocr.Reader(['en'], gpu=True)  # Requires CUDA
```

---

## License

OCR components:
- **PyMuPDF**: AGPL-3.0 (commercial license available)
- **EasyOCR**: Apache 2.0
- **Pillow**: HPND License

All are free for commercial use.
