# Lecture Video Question Answering Using RAG

A multimodal Retrieval-Augmented Generation (RAG) system that lets you upload a lecture video and ask natural-language questions about its content. The pipeline combines **visual text on screen (OCR)** with **spoken content (speech-to-text)** to build a searchable knowledge base of the lecture, then uses an LLM to generate grounded answers.

## How it works

```
                         ┌───────────────────┐
                         │   Upload Video     │
                         └─────────┬──────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                          ▼
  ┌───────────────────────┐                 ┌───────────────────────────┐
  │ 1. Frame Extraction    │                 │ 3. Audio Transcription    │
  │  (frame_extractor.py)  │                 │  (audio_transcriber.py)   │
  │  • samples frames at   │                 │  • extracts audio track   │
  │    an adaptive interval│                 │  • transcribes with       │
  │  • drops near-duplicate│                 │    OpenAI Whisper         │
  │    frames               │                 │  • timestamped segments   │
  └───────────┬─────────────┘                 └─────────────┬─────────────┘
              ▼                                              │
  ┌───────────────────────┐                                  │
  │ 2. OCR                │                                  │
  │  (ocr_processor.py)   │                                  │
  │  • reads on-screen text│                                 │
  │    (slides/whiteboard)│                                  │
  │    with EasyOCR        │                                 │
  └───────────┬─────────────┘                                │
              └───────────────────┬──────────────────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │ 4. Chunk Merging           │
                     │  (chunk_merger.py)         │
                     │  • aligns OCR text with the │
                     │    transcript by timestamp  │
                     │  • builds ~20s text chunks  │
                     └─────────────┬───────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │ 5. Embedding + Indexing    │
                     │  (embedder.py)             │
                     │  • Sentence-Transformers    │
                     │    (all-mpnet-base-v2)      │
                     │  • FAISS vector index       │
                     └─────────────┬───────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │ 6. Question Answering       │
                     │  (qa_engine.py)             │
                     │  • embeds the question       │
                     │  • retrieves top-k chunks    │
                     │  • Llama-3-8B-Instruct        │
                     │    (via HF Inference API)     │
                     │    generates the answer       │
                     └───────────────────────────┘
```

All of this is wired together in `app.py`, a **Streamlit** app that lets you upload a video, run the pipeline step by step with progress feedback, and then ask questions in a simple chat-like box.

## Project structure

```
lecture-video-qa-rag/
├── app.py                     # Streamlit UI and pipeline orchestration
├── pipeline/
│   ├── frame_extractor.py     # Step 1: sample & de-duplicate video frames
│   ├── ocr_processor.py       # Step 2: OCR on extracted frames (EasyOCR)
│   ├── audio_transcriber.py   # Step 3: speech-to-text (Whisper)
│   ├── chunk_merger.py        # Step 4: merge OCR + transcript into chunks
│   ├── embedder.py            # Step 5: embed chunks & build FAISS index
│   └── qa_engine.py           # Step 6: retrieval + LLM answer generation
├── utils/
│   └── helpers.py             # Shared JSON I/O & text-cleaning helpers
├── sessions/                  # Per-upload working data (gitignored)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/lecture-video-qa-rag.git
cd lecture-video-qa-rag

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Note:** `easyocr`, `openai-whisper`, and `sentence-transformers` will download their model weights on first run. This may take a few minutes and needs a working internet connection.

`moviepy` (used for audio extraction) also requires **ffmpeg** to be installed on your system:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg

# Windows: https://ffmpeg.org/download.html
```

### 2. Configure your Hugging Face token

The QA engine calls `meta-llama/Meta-Llama-3-8B-Instruct` through the Hugging Face Inference API.

1. Create a token at https://huggingface.co/settings/tokens
2. Accept the model's license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
3. Copy the example env file and add your token:

```bash
cp .env.example .env
# then edit .env and set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### 3. Run the app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`), upload a lecture video, click **Process Video**, and once processing finishes, ask questions in the **Ask Questions** box.

## Pipeline details

| Stage | File | What it does |
|---|---|---|
| Frame extraction | `pipeline/frame_extractor.py` | Samples frames at an interval that adapts to video length (5s / 7s / 10s) and skips near-duplicate frames using pixel-difference thresholding. |
| OCR | `pipeline/ocr_processor.py` | Runs EasyOCR on each sampled frame, keeping only text with confidence > 0.7 and length > 3 characters; tracks the previous frame's OCR for continuity. |
| Audio transcription | `pipeline/audio_transcriber.py` | Extracts the audio track with MoviePy and transcribes it with OpenAI Whisper (`base` model), producing timestamped segments. |
| Chunk merging | `pipeline/chunk_merger.py` | Deduplicates repeated OCR text, aligns it with transcript segments by timestamp, and groups everything into ~20-second text chunks. |
| Embedding | `pipeline/embedder.py` | Encodes each chunk with `all-mpnet-base-v2` and stores the vectors in a FAISS flat inner-product index, alongside chunk metadata. |
| QA | `pipeline/qa_engine.py` | Embeds the incoming question, retrieves the top-6 most similar chunks from FAISS, and prompts Llama-3-8B-Instruct to produce a clean 2–4 sentence answer grounded in that context. |

## Known limitations

- Processing time scales with video length — frame extraction, OCR, and Whisper transcription all run on CPU by default unless you have CUDA set up for PyTorch.
- The Whisper `base` model favors speed over accuracy; swap in `small` or `medium` in `audio_transcriber.py` for better transcripts at the cost of speed.
- Session data (frames, transcripts, indexes) is stored on local disk under `sessions/<session_id>/` and is **not** automatically cleaned up unless you click "Delete Session Data" in the app.
- The Hugging Face Inference API has rate limits on free-tier tokens; heavy use may require a paid Inference Endpoint.

## Roadmap ideas

- [ ] Swap FAISS flat index for an approximate index (e.g. HNSW) for larger lecture libraries
- [ ] Add support for multiple videos in one session with cross-video search
- [ ] Cache embeddings to avoid recomputation when re-processing the same video
- [ ] Add automated tests for each pipeline stage

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
