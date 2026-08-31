# ==========================================
# Frame Extraction Pipeline
# ==========================================

import cv2
import os

from utils.helpers import create_folder, save_json


def extract_frames(video_path, output_base_path):

    video_name = os.path.basename(video_path)
    video_id = os.path.splitext(video_name)[0]

    frame_folder = os.path.join(
        output_base_path,
        video_id,
        "frames"
    )

    metadata_path = os.path.join(
        output_base_path,
        video_id,
        "frame_metadata.json"
    )

    create_folder(frame_folder)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_name}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration_sec = total_frames / fps

    # Sampling logic
    if duration_sec <= 600:
        interval_sec = 5
    elif duration_sec <= 1200:
        interval_sec = 7
    else:
        interval_sec = 10

    frame_interval = int(fps * interval_sec)

    frame_id = 0
    saved_id = 0

    frame_data = []

    prev_gray = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % frame_interval == 0:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Duplicate filtering
            if prev_gray is not None:

                diff = cv2.absdiff(prev_gray, gray)

                non_zero_count = cv2.countNonZero(diff)

                if non_zero_count < 500:
                    frame_id += 1
                    continue

            prev_gray = gray

            timestamp = frame_id / fps

            frame_name = os.path.join(
                frame_folder,
                f"frame_{saved_id}.jpg"
            )

            cv2.imwrite(frame_name, frame)

            frame_data.append({
                "frame": frame_name,
                "time": round(timestamp, 2)
            })

            saved_id += 1

        frame_id += 1

    cap.release()

    save_json(metadata_path, frame_data)

    return {
    "metadata_path": metadata_path,
    "total_frames": saved_id
}