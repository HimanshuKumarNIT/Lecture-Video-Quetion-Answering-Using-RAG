# ==========================================
# OCR Processing Pipeline
# ==========================================

import os
import cv2
import easyocr

from utils.helpers import (
    load_json,
    save_json,
    clean_ocr
)

# Load OCR model once
reader = easyocr.Reader(['en'])


def smart_resize(img, max_width=1000):

    h, w = img.shape[:2]

    if w > max_width:

        scale = max_width / w

        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale))
        )

    return img


def process_ocr(output_base_path, video_id):

    input_path = os.path.join(
        output_base_path,
        video_id,
        "frame_metadata.json"
    )

    output_path = os.path.join(
        output_base_path,
        video_id,
        "frame_with_ocr.json"
    )

    if not os.path.exists(input_path):
        raise Exception("Frame metadata not found")

    frames = load_json(input_path)

    updated = []

    prev_ocr = ""

    for item in frames:

        img_path = item["frame"]

        img = cv2.imread(img_path)

        if img is None:

            item["ocr"] = ""
            item["prev_ocr"] = ""

            updated.append(item)

            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = smart_resize(gray)

        # OCR
        results = reader.readtext(gray)

        texts = []

        for (_, text, prob) in results:

            if prob > 0.7 and len(text.strip()) > 3:

                texts.append(clean_ocr(text))

        current_ocr = " ".join(texts)

        # Context continuity
        item["ocr"] = current_ocr
        item["prev_ocr"] = prev_ocr

        prev_ocr = current_ocr

        updated.append(item)

    save_json(output_path, updated)

    # Sample OCR for UI preview
    sample_ocr = ""

    for item in updated:

        if item.get("ocr"):

            sample_ocr = item["ocr"][:300]

            break

    return {
        "output_path": output_path,
        "sample_ocr": sample_ocr
    }