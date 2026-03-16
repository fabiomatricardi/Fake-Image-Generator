# starting point is:
# dataset from kaupane/nano-banana-pro-gen
# json prepared from  1.big_dset_parquet_Ask.py  and saved in kaupane_nano-banana-pro-gen.json
# at first run it will detect no FAISS db and ask for creation
# requires llama-cpp server running with bge-m3-q8_0.gguf that is multilingual
# .\llama-server.exe -m E:\EmbeddingModels\bge-m3-q8_0.gguf --embedding --ctx-size 512 --batch-size 512

import gradio as gr
import json
import requests
from PIL import Image
import os
import faiss
import numpy as np
import random
from typing import List, Tuple, Dict
import time

# ============ CONFIGURATION ============
LLAMA_CPP_SERVER_URL = "http://localhost:8080"
EMBEDDING_ENDPOINT = f"{LLAMA_CPP_SERVER_URL}/embedding"
DATABASE_FILE = "cerealt_open-image-preferences-v1-binarized.json"   #our new dataset
IMAGES_FOLDER = "./images"
TOP_K = 5
FAISS_INDEX_FILE = "faiss_index.bin"
FAISS_METADATA_FILE = "faiss_metadata.json"
EMBEDDING_DIM = 1024

# ============ EMBEDDING FUNCTIONS ============
def get_embedding(text: str) -> np.ndarray:
    try:
        response = requests.post(
            EMBEDDING_ENDPOINT,
            json={"content": text},
            timeout=30
        )
        response.raise_for_status()
        response_data = response.json()

        if isinstance(response_data, list) and len(response_data) > 0:
            first_item = response_data[0]
            if 'embedding' in first_item:
                embedding_data = first_item['embedding']
                if isinstance(embedding_data, list) and isinstance(embedding_data[0], list):
                    embedding_data = embedding_data[0]
                embedding = np.array(embedding_data, dtype=np.float32)
        elif isinstance(response_data, dict):
            if "embedding" in response:
                embedding = np.array(response_data["embedding"], dtype=np.float32)
            elif "data" in response_data and len(response_data["data"]) > 0:
                embedding = np.array(response_data["data"][0]["embedding"], dtype=np.float32)
            else:
                raise Exception("Unexpected response format")
        else:
            raise Exception("Unexpected response type")

        return embedding / np.linalg.norm(embedding)

    except Exception as e:
        raise Exception(f"Failed to get embedding: {e}")

# ============ FAISS INDEX FUNCTIONS ============
def build_faiss_index(database: List[Dict], progress=gr.Progress()) -> faiss.IndexFlatIP:
    print(f"\n🔨 Building FAISS index for {len(database)} entries...")
    progress(0, desc="Computing embeddings...")
    embeddings = []
    valid_indices = []

    for i, entry in enumerate(database):
        try:
            prompt = entry.get('prompt', '')
            if prompt:
                embedding = get_embedding(prompt)
                embeddings.append(embedding)
                valid_indices.append(i)
                if (i + 1) % 50 == 0:
                    progress((i + 1) / len(database), desc=f"Computing... ({i+1}/{len(database)})")
        except Exception as e:
            print(f"⚠️ Error at index {i}: {e}")
            continue

    if not embeddings:
        raise Exception("No valid embeddings computed!")

    embeddings_array = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings_array)

    faiss.write_index(index, FAISS_INDEX_FILE)
    metadata = {
        'valid_indices': valid_indices,
        'total_entries': len(database),
        'embedding_dim': EMBEDDING_DIM,
        'database_file': DATABASE_FILE
    }
    with open(FAISS_METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    progress(1.0, desc="Index built successfully!")
    return index

def load_faiss_index() -> Tuple[faiss.IndexFlatIP, Dict]:
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_FILE}")
    if not os.path.exists(FAISS_METADATA_FILE):
        raise FileNotFoundError(f"FAISS metadata not found: {FAISS_METADATA_FILE}")
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(FAISS_METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return index, metadata

def load_database(db_file: str) -> List[Dict]:
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"Database file not found: {db_file}")
    with open(db_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# ============ SEARCH FUNCTION (NOW A GENERATOR) ============
def perform_search(query: str, progress=gr.Progress()):
    """Generator function: yields noise frames, then final results."""
    # Yield noise frames first
    for _ in range(10):  # Fixed 10 steps of noise
        time.sleep(0.21)
        noise = np.random.random((640, 640, 3))  # RGB noise with aspect ratio 1:1
        yield noise, None, None  # Update only the image during noise phase

    # After noise, do the real search and yield final results
    if not query.strip():
        empty_gallery = [None] * 4
        empty_details = "❌ Please enter a search query"
        yield None, empty_gallery, empty_details
        return

    try:
        database = load_database(DATABASE_FILE)
        index, metadata = load_faiss_index()
        image_paths, prompts, scores = search_with_faiss(query, index, metadata, database, TOP_K)

        best_image = None
        gallery_results = []
        detailed_html_str = ""

        if len(image_paths) > 0 and image_paths[0] and os.path.exists(image_paths[0]):
            best_image = Image.open(image_paths[0])
            detailed_html_str = f"<div><strong>Best Match - Score: {scores[0]:.4f}</strong><br>Prompt: {prompts[0]}</div>"

        for i in range(1, TOP_K):
            if i < len(image_paths) and image_paths[i] and os.path.exists(image_paths[i]):
                img = Image.open(image_paths[i])
                gallery_results.append((img, f"#{i+1} - {scores[i]:.4f}"))
                detailed_html_str += f"<div><strong>Match #{i+1} - {scores[i]:.4f}</strong>: {prompts[i][:100]}...</div>"
            else:
                gallery_results.append((None, "Not found"))

        yield best_image, gallery_results, detailed_html_str

    except Exception as e:
        empty_gallery = [None] * 4
        error_msg = f"❌ Error: {str(e)}"
        yield None, empty_gallery, error_msg

# ============ SIMPLE SEARCH FUNCTION FOR FAISS ============
def search_with_faiss(query: str, index: faiss.IndexFlatIP, metadata: Dict, database: List[Dict], top_k: int = 5):
    """Search for similar images using FAISS."""
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    # Get embedding for the query
    query_embedding = get_embedding(query).reshape(1, -1)
    
    # Perform search
    scores, indices = index.search(query_embedding, top_k)

    image_paths = []
    prompts = []
    final_scores = []
    valid_indices = metadata['valid_indices']

    for score, faiss_idx in zip(scores[0], indices[0]):
        if faiss_idx == -1:
            continue
        db_idx = valid_indices[faiss_idx]
        entry = database[db_idx]
        img_path = os.path.join(IMAGES_FOLDER, os.path.basename(entry.get('filename', '')))
        image_paths.append(img_path if os.path.exists(img_path) else None)
        prompts.append(entry.get('prompt', 'No prompt'))
        final_scores.append(float(score))

    return image_paths, prompts, final_scores

# ============ UI & STATUS FUNCTIONS ============
def check_index_status() -> str:
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_METADATA_FILE):
        try:
            index = faiss.read_index(FAISS_INDEX_FILE)
            return f"✅ Ready: {index.ntotal:,} vectors"
        except Exception as e:
            return f"⚠️ Corrupted: {e}"
    else:
        return "❌ Index not built"

def create_interface():
    with gr.Blocks(title="Fake Image Generation with Noise Effect") as demo:
        gr.Markdown("# 🖼️ Fake Image Generation")
        gr.Markdown("#### 👾 with diffusion noise effect")

        with gr.Row():
            status_output = gr.Textbox(label="Status", value=check_index_status(), interactive=False)

        with gr.Row():
            query_input = gr.Textbox(
                label="Enter Prompt (English)",
                placeholder="e.g., space station, tank in battlefield...",
                lines=2
            )

        search_btn = gr.Button("Generate", variant="primary")
        
        # Output components
        best_match_image = gr.Image(
            label="Generated Image",
            streaming=True,  # Enable streaming for generator
            height=486
        )
        results_gallery = gr.Gallery(label="Other Matches", 
                columns=2, 
                object_fit="contain",
                height="auto",
                allow_preview=True,
                preview=True,
                scale=1,
                rows=2,
                selected_index=0)
        detailed_results_html = gr.HTML(label="Details")

        # Examples
        gr.Examples(
            examples=[
                "space station with solar panels",
                "military tank on battlefield",
                "ship in fog and snow"
            ],
            inputs=query_input
        )

        # Event handler
        search_btn.click(
            fn=perform_search,
            inputs=[query_input],
            outputs=[best_match_image, results_gallery, detailed_results_html]
        )

        query_input.submit(
            fn=perform_search,
            inputs=[query_input],
            outputs=[best_match_image, results_gallery, detailed_results_html]
        )

        # Refresh status
        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
        refresh_btn.click(fn=check_index_status, outputs=status_output)

    return demo

# ============ MAIN ============
if __name__ == "__main__":
    print("="*50)
    print("🚀 Starting Fake Image Generator with Noise Effect")
    print("="*50)

    demo = create_interface()
    demo.launch(inbrowser=True, server_port=7860)