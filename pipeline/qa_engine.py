# ==========================================
# Question Answering Engine
# ==========================================

import os
import json
import faiss
import numpy as np

from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

from huggingface_hub import InferenceClient


# Load .env
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise Exception("HF_TOKEN not found in .env")


# Embedding model
embed_model = SentenceTransformer(
    'all-mpnet-base-v2'
)

# LLaMA client
client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=HF_TOKEN
)


def generate_answer_llama(context, question):

    response = client.chat_completion(

        messages=[

            {
                "role": "system",

                "content": (
                    "You are an expert tutor. "
                    "Your job is to understand messy lecture notes "
                    "and generate clear, correct, and well-structured explanations.\n\n"

                    "STRICT RULES:\n"
                    "- Do NOT copy sentences from context\n"
                    "- Fix incorrect, noisy, or broken words\n"
                    "- Rewrite everything in clean English\n"
                    "- Focus on explaining the concept clearly\n"
                    "- Avoid unnecessary repetition\n"
                    "- Answer must be 2 to 4 meaningful sentences\n"
                    "- Output should look like a human explanation"
                )
            },

            {
                "role": "user",

                "content": f"""
Lecture Context:
{context}

Question:
{question}

Now generate a clean, well-explained answer.
"""
            }
        ],

        max_tokens=180,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


def answer_question(output_base_path, video_id, question):

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

    if not os.path.exists(index_path):
        raise Exception("FAISS index not found")

    if not os.path.exists(meta_path):
        raise Exception("Meta file not found")

    # Load FAISS
    index = faiss.read_index(index_path)

    # Load metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Query embedding
    q_vec = embed_model.encode(
        [question],
        normalize_embeddings=True
    )

    q_vec = np.array(q_vec).astype("float32")

    # Search
    _, idxs = index.search(q_vec, 6)

    # Context building
    context = " ".join([

        chunks[i]["text"]

        for i in idxs[0]

        if i < len(chunks)
    ])

    # Generate answer
    answer_text = generate_answer_llama(
        context,
        question
    )

    return answer_text