"""
📌 Script: Build FAISS Index for Fake Image Generation
📝 Purpose: Creates a semantic search index from your JSON dataset
✅ Run this ONCE before starting your Gradio app
--> adapted with a new dataset from Hugging Face 
    https://huggingface.co/datasets/cerealt/open-image-preferences-v1-binarized
"""

import os
import json
import numpy as np
import faiss
import requests
from typing import List, Dict

# ============ CONFIGURATION ============
LLAMA_CPP_SERVER_URL = "http://localhost:8080"
EMBEDDING_ENDPOINT = f"{LLAMA_CPP_SERVER_URL}/embedding"

# DATABASE_FILE = "kaupane_nano-banana-pro-gen.json"  
# change THIS! with "cerealt_open-image-preferences-v1-binarized.json"
DATABASE_FILE = "cerealt_open-image-preferences-v1-binarized.json"  
IMAGES_FOLDER = "./images"

FAISS_INDEX_FILE = "faiss_index.bin"
FAISS_METADATA_FILE = "faiss_metadata.json"
EMBEDDING_DIM = 1024  # bge-m3 embedding dimension

# Number of results to retrieve
TOP_K = 5


# ============ EMBEDDING FUNCTION ============
def get_embedding(text: str) -> np.ndarray:
    """Get embedding from local llama.cpp server"""
    try:
        response = requests.post(
            EMBEDDING_ENDPOINT,
            json={"content": text},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list) and len(data) > 0:
            embedding_data = data[0]["embedding"]
            if isinstance(embedding_data[0], list):  # Handle nested format
                embedding_data = embedding_data[0]
        elif isinstance(data, dict):
            if "embedding" in data:
                embedding_data = data["embedding"]
            elif "data" in data and len(data["data"]) > 0:
                embedding_data = data["data"][0]["embedding"]
            else:
                raise ValueError("No embedding found in response")
        else:
            raise ValueError(f"Unexpected response type: {type(data)}")

        embedding = np.array(embedding_data, dtype=np.float32)
        return embedding / np.linalg.norm(embedding)  # Normalize

    except Exception as e:
        print(f"❌ Failed to get embedding for text: '{text[:50]}...'")
        raise Exception(f"Embedding error: {e}")


# ============ LOAD DATASET ============
def load_dataset() -> List[Dict]:
    """Load the JSON dataset"""
    if not os.path.exists(DATABASE_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATABASE_FILE}")
    with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} entries from {DATABASE_FILE}")
    return data


# ============ BUILD FAISS INDEX ============
def build_index():
    print("🚀 Starting FAISS index build...")
    print(f"   Dataset: {DATABASE_FILE}")
    print(f"   Output : {FAISS_INDEX_FILE}, {FAISS_METADATA_FILE}")
    print("-" * 60)

    # Step 1: Load dataset
    try:
        database = load_dataset()
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Step 2: Test connection to llama.cpp
    try:
        response = requests.get(f"{LLAMA_CPP_SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            if health.get("status") == "ok":
                print(f"✅ Connected to llama.cpp server ({LLAMA_CPP_SERVER_URL})")
            else:
                print(f"❌ Unexpected health response: {health}")
                return
        else:
            print(f"❌ Health check failed with status code: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to llama.cpp server!")
        print(f"💡 Please start it with:")
        print(f"   ./llama-server.exe -m E:\\EmbeddingModels\\bge-m3-q8_0.gguf --embedding --port 8080 --ctx-size 512")
        return
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return

    # Step 3: Compute embeddings
    print(f"\n🔨 Computing embeddings for prompts (dim={EMBEDDING_DIM})...")
    embeddings = []
    valid_indices = []

    for i, entry in enumerate(database):
        prompt = entry.get('prompt', '').strip()
        if not prompt:
            continue

        try:
            embedding = get_embedding(prompt)
            embeddings.append(embedding)
            valid_indices.append(i)
            if (i + 1) % 10 == 0:
                print(f"   Processed {i+1}/{len(database)} entries...")

        except Exception as e:
            print(f"⚠️ Skipped entry {i}: {e}")
            continue

    if len(embeddings) == 0:
        print("❌ No valid embeddings computed! Check your server or dataset.")
        return

    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    print(f"✅ Computed {len(embeddings_array)} embeddings")

    # Step 4: Create and populate FAISS index
    print("\n🧠 Creating FAISS IndexFlatIP index...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings_array)
    print(f"✅ Added {index.ntotal} vectors to index")

    # Save index
    print(f"\n💾 Saving FAISS index to disk...")
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"   Index saved: {FAISS_INDEX_FILE}")

    # Save metadata
    metadata = {
        'valid_indices': valid_indices,
        'total_entries': len(database),
        'embedding_dim': EMBEDDING_DIM,
        'database_file': DATABASE_FILE,
        'vectors_stored': len(embeddings_array)
    }
    with open(FAISS_METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   Metadata saved: {FAISS_METADATA_FILE}")

    # Final summary
    print("\n" + "="*60)
    print("🎉 FAISS INDEX BUILT SUCCESSFULLY!")
    print("="*60)
    print(f"📊 Vectors indexed : {index.ntotal:,}")
    print(f"📁 Total dataset   : {len(database):,} entries")
    print(f"✅ Valid prompts   : {len(valid_indices):,}")
    print(f"🔍 Top-K Results   : {TOP_K}")
    print(f"⚡ You can now run your Gradio app!")


# ============ MAIN ============
if __name__ == "__main__":
    build_index()