# ==========================================
# Embedding + FAISS Pipeline
# ==========================================

import os
import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

# Load embedding model once
embed_model = SentenceTransformer(
    'all-mpnet-base-v2'
)


def create_embeddings(output_base_path, video_id):

    input_path = os.path.join(
        output_base_path,
        video_id,
        "final_chunks.json"
    )

    index_path = os.path.join(
        output_base_path,
        video_id,
        "faiss.index"
    )

    meta_path = os.path.join(
        output_base_path,
        video_id,
        "meta.json"
    )

    if not os.path.exists(input_path):
        raise Exception("Final chunks not found")

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [

        chunk["text"]

        for chunk in chunks

        if chunk["text"].strip()
    ]

    if len(texts) == 0:
        raise Exception("No valid text found")

    # Generate embeddings
    embeddings = embed_model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    # Save FAISS
    faiss.write_index(index, index_path)

    # Save metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    return {
    "index_path": index_path,
    "embedding_count": len(texts)
}