# ==========================================
# Audio Transcription Pipeline
# ==========================================

import os
import whisper

from moviepy.editor import VideoFileClip

from utils.helpers import (
    save_json,
    clean_text
)

# Load whisper model once
whisper_model = whisper.load_model("base")


def transcribe_audio(video_path, output_base_path):

    video_name = os.path.basename(video_path)

    video_id = os.path.splitext(video_name)[0]

    audio_path = os.path.join(
        output_base_path,
        video_id,
        "audio.wav"
    )

    transcript_path = os.path.join(
        output_base_path,
        video_id,
        "transcript.json"
    )

    # Extract audio
    video = VideoFileClip(video_path)

    video.audio.write_audiofile(
        audio_path,
        verbose=False,
        logger=None
    )

    # Whisper transcription
    result = whisper_model.transcribe(
        audio_path,
        fp16=False,
        task="transcribe"
    )

    segments = result["segments"]

    cleaned_segments = []

    for seg in segments:

        cleaned_segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": clean_text(seg["text"])
        })

    save_json(transcript_path, cleaned_segments)

    # Sample transcript for UI preview
    sample_text = ""

    if len(cleaned_segments) > 0:

        sample_text = cleaned_segments[0]["text"]

    return {
        "transcript_path": transcript_path,
        "sample_transcript": sample_text
    }