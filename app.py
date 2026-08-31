# ==========================================
# Streamlit App
# ==========================================

import streamlit as st
import os
import uuid
import shutil

from pipeline.frame_extractor import extract_frames
from pipeline.ocr_processor import process_ocr
from pipeline.audio_transcriber import transcribe_audio
from pipeline.chunk_merger import merge_chunks
from pipeline.embedder import create_embeddings
from pipeline.qa_engine import answer_question


# ==========================================
# BASE SESSION FOLDER
# ==========================================

SESSION_BASE = "sessions"

os.makedirs(SESSION_BASE, exist_ok=True)


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Multimodal Video QA",
    layout="wide"
)

st.title("Multimodal Video Question Answering")

st.write(
    "Upload a lecture video and ask questions from it."
)


# ==========================================
# SESSION SETUP
# ==========================================

if "session_id" not in st.session_state:

    st.session_state.session_id = str(uuid.uuid4())[:8]

session_id = st.session_state.session_id

session_folder = os.path.join(
    SESSION_BASE,
    session_id
)

os.makedirs(session_folder, exist_ok=True)


# ==========================================
# VIDEO UPLOAD
# ==========================================

uploaded_video = st.file_uploader(
    "Upload Video",
    type=["mp4", "mov", "avi"]
)


if uploaded_video is not None:

    video_path = os.path.join(
        session_folder,
        uploaded_video.name
    )

    # Save uploaded video
    with open(video_path, "wb") as f:

        f.write(uploaded_video.read())

    st.success("Video uploaded successfully")

    video_id = os.path.splitext(
        uploaded_video.name
    )[0]

    st.info(f"Session ID: {session_id}")


    # ==========================================
    # PROCESS VIDEO
    # ==========================================

    if st.button("Process Video"):

        progress_bar = st.progress(0)

        # ==========================================
        # STEP 1: FRAME EXTRACTION
        # ==========================================

        try:

            st.info("STEP 1: Extracting frames...")

            frame_result = extract_frames(
                video_path,
                session_folder
            )

            progress_bar.progress(20)

            st.success("Frame extraction completed")

            st.write(
                f"Frames Extracted: {frame_result['total_frames']}"
            )

            st.code(
                frame_result["metadata_path"]
            )

        except Exception as e:

            st.error(
                f"Frame Extraction Failed: {e}"
            )

            st.stop()


        # ==========================================
        # STEP 2: OCR
        # ==========================================

        try:

            st.info("STEP 2: Running OCR...")

            ocr_result = process_ocr(
                session_folder,
                video_id
            )

            progress_bar.progress(40)

            st.success("OCR completed")

            st.write("Sample OCR Output:")

            st.code(
                ocr_result["sample_ocr"]
            )

        except Exception as e:

            st.error(
                f"OCR Failed: {e}"
            )

            st.stop()


        # ==========================================
        # STEP 3: AUDIO TRANSCRIPTION
        # ==========================================

        try:

            st.info("STEP 3: Transcribing audio...")

            transcript_result = transcribe_audio(
                video_path,
                session_folder
            )

            progress_bar.progress(60)

            st.success(
                "Audio transcription completed"
            )

            st.write("Sample Transcript:")

            st.code(
                transcript_result["sample_transcript"]
            )

        except Exception as e:

            st.error(
                f"Audio Transcription Failed: {e}"
            )

            st.stop()


        # ==========================================
        # STEP 4: CHUNK MERGING
        # ==========================================

        try:

            st.info("STEP 4: Merging chunks...")

            chunk_result = merge_chunks(
                session_folder,
                video_id
            )

            progress_bar.progress(80)

            st.success(
                "Chunk merging completed"
            )

            st.write(
                f"Chunks Created: {chunk_result['total_chunks']}"
            )

        except Exception as e:

            st.error(
                f"Chunk Merging Failed: {e}"
            )

            st.stop()


        # ==========================================
        # STEP 5: EMBEDDING CREATION
        # ==========================================

        try:

            st.info("STEP 5: Creating embeddings...")

            embedding_result = create_embeddings(
                session_folder,
                video_id
            )

            progress_bar.progress(100)

            st.success(
                "Embeddings + FAISS created"
            )

            st.write(
                f"Embeddings Created: {embedding_result['embedding_count']}"
            )

        except Exception as e:

            st.error(
                f"Embedding Failed: {e}"
            )

            st.stop()


        # ==========================================
        # FINISHED
        # ==========================================

        st.balloons()

        st.success(
            "Full pipeline completed successfully!"
        )


    # ==========================================
    # QUESTION ANSWERING
    # ==========================================

    st.subheader("Ask Questions")

    question = st.text_input(
        "Enter your question"
    )

    if st.button("Get Answer"):

        if question.strip() == "":

            st.warning(
                "Please enter a valid question"
            )

        else:

            try:

                with st.spinner(
                    "Generating answer..."
                ):

                    ans = answer_question(
                        session_folder,
                        video_id,
                        question
                    )

                st.success(
                    "Answer Generated"
                )

                st.write(ans)

            except Exception as e:

                st.error(f"Error: {e}")


    # ==========================================
    # SESSION FILE VIEWER
    # ==========================================

    st.subheader("Session Files")

    session_video_folder = os.path.join(
        session_folder,
        video_id
    )

    if os.path.exists(session_video_folder):

        all_files = []

        for root, dirs, files in os.walk(
            session_video_folder
        ):

            for file in files:

                full_path = os.path.join(
                    root,
                    file
                )

                relative_path = os.path.relpath(
                    full_path,
                    session_folder
                )

                all_files.append(
                    relative_path
                )

        if len(all_files) > 0:

            st.write("Generated Files:")

            for file in all_files:

                st.code(file)

        else:

            st.info(
                "No generated files yet"
            )


    # ==========================================
    # CLEANUP SESSION
    # ==========================================

    st.subheader("🗑 Cleanup Session")

    if st.button("Delete Session Data"):

        try:

            shutil.rmtree(session_folder)

            st.success(
                "Session data deleted successfully"
            )

            del st.session_state.session_id

        except Exception as e:

            st.error(
                f"Cleanup Error: {e}"
            )