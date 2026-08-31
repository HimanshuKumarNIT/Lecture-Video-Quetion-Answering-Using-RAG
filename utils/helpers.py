# ==========================================
# Shared Utility Helpers
# ==========================================
#
# Common file I/O and text-cleaning helpers used across the
# frame extraction, OCR, transcription, chunking, and embedding
# stages of the pipeline.

import os
import json
import re


def create_folder(path):
    """Create a folder (and any missing parent folders) if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_json(path):
    """Load and return JSON data from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    """Save data as JSON to a file, creating parent folders if needed."""
    create_folder(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clean_text(text):
    """
    Basic cleanup for Whisper transcript segments:
    - collapse extra whitespace
    - strip leading/trailing spaces
    """
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_ocr(text):
    """
    Cleanup for noisy OCR output:
    - collapse whitespace
    - drop stray non-alphanumeric junk characters commonly
      produced by OCR on slides/whiteboards
    - strip leading/trailing spaces
    """
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"[^a-zA-Z0-9.,:;()%+\-/=\s]", "", text)

    return text.strip()
