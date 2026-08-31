# ==========================================
# Chunk Merging Pipeline
# ==========================================

import os

from utils.helpers import (
    load_json,
    save_json
)

CHUNK_WINDOW = 20


def merge_chunks(output_base_path, video_id):

    frame_path = os.path.join(
        output_base_path,
        video_id,
        "frame_with_ocr.json"
    )

    audio_path = os.path.join(
        output_base_path,
        video_id,
        "transcript.json"
    )

    output_path = os.path.join(
        output_base_path,
        video_id,
        "final_chunks.json"
    )

    if not os.path.exists(frame_path):
        raise Exception("OCR file not found")

    if not os.path.exists(audio_path):
        raise Exception("Transcript file not found")

    frames = load_json(frame_path)

    audio = load_json(audio_path)

    # Remove redundant OCR frames
    filtered_frames = []

    prev_ocr = ""

    for frame in frames:

        curr_ocr = frame.get("ocr", "")

        if curr_ocr and curr_ocr != prev_ocr:

            filtered_frames.append(frame)

            prev_ocr = curr_ocr

    # Merge OCR + audio
    chunks = []

    current_text = []

    start_time = 0

    for segment in audio:

        seg_start = segment["start"]

        seg_end = segment["end"]

        seg_text = segment["text"]

        if not current_text:
            start_time = seg_start

        # OCR matching by timestamp
        ocr_texts = [

            frame["ocr"]

            for frame in filtered_frames

            if seg_start <= frame["time"] <= seg_end
            and frame.get("ocr")
        ]

        combined = f"{seg_text} {' '.join(ocr_texts)}"

        current_text.append(combined)

        # Chunk creation
        if seg_end - start_time >= CHUNK_WINDOW:

            chunks.append({
                "start": round(start_time, 2),
                "end": round(seg_end, 2),
                "text": " ".join(current_text)
            })

            current_text = []

    # Last chunk
    if current_text:

        chunks.append({
            "start": round(start_time, 2),
            "end": round(audio[-1]["end"], 2),
            "text": " ".join(current_text)
        })

    save_json(output_path, chunks)

    return {
    "output_path": output_path,
    "total_chunks": len(chunks)
}