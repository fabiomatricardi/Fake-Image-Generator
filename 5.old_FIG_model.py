# starting point is:
# dataset from kaupane/nano-banana-pro-gen
# json prepared from  1.big_dset_parquet_Ask.py  and saved in kaupane_nano-banana-pro-gen.json
# at first run it will detect no FAISS db and ask for creation
# requires llama-cpp server running with bge-m3-q8_0.gguf that is multilingual
# .\llama-server.exe -m E:\EmbeddingModels\bge-m3-q8_0.gguf --embedding --ctx-size 512 --batch-size 512

import gradio as gr
import json
import numpy as np
import requests
from PIL import Image, ImageFilter
import os
from pathlib import Path
from typing import List, Tuple, Dict
import time
import faiss
import base64
from io import BytesIO

# ============ CONFIGURATION ============
LLAMA_CPP_SERVER_URL = "http://localhost:8080"  # llama.cpp server endpoint
EMBEDDING_ENDPOINT = f"{LLAMA_CPP_SERVER_URL}/embedding"
DATABASE_FILE = "kaupane_nano-banana-pro-gen.json"  # Your JSON database
IMAGES_FOLDER = "./images"
TOP_K = 5  # Changed from 4 to 5
ANIMATION_DURATION = 5  # seconds
ANIMATION_FRAMES = 20  # Number of blur levels

# FAISS index files
FAISS_INDEX_FILE = "faiss_index.bin"
FAISS_METADATA_FILE = "faiss_metadata.json"
EMBEDDING_DIM = 1024  # bge-m3 embedding dimension

# Cache for blurred images
BLUR_CACHE_FOLDER = "./blur_cache"
os.makedirs(BLUR_CACHE_FOLDER, exist_ok=True)


# ============ EMBEDDING FUNCTIONS ============
def get_embedding(text: str) -> np.ndarray:
    """Get embedding from llama.cpp server"""
    try:
        response = requests.post(
            EMBEDDING_ENDPOINT,
            json={"content": text},
            timeout=30
        )
        response.raise_for_status()
        response_data = response.json()
        
        # Handle llama.cpp embedding format: [{'index': 0, 'embedding': [[...]]}]
        if isinstance(response_data, list) and len(response_data) > 0:
            first_item = response_data[0]
            if isinstance(first_item, dict) and 'embedding' in first_item:
                embedding_data = first_item['embedding']
                
                # Flatten nested list if needed
                if isinstance(embedding_data, list) and len(embedding_data) > 0:
                    if isinstance(embedding_data[0], list):
                        embedding_data = embedding_data[0]
                
                embedding = np.array(embedding_data, dtype=np.float32)
            else:
                raise Exception(f"Unexpected item format: {first_item}")
        elif isinstance(response_data, dict):
            if "embedding" in response_data:
                embedding = np.array(response_data["embedding"], dtype=np.float32)
            elif "data" in response_data and len(response_data["data"]) > 0:
                embedding = np.array(response_data["data"][0]["embedding"], dtype=np.float32)
            else:
                raise Exception(f"Unexpected response format")
        else:
            raise Exception(f"Unexpected response type: {type(response_data)}")
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to get embedding from llama.cpp server: {e}")
    except Exception as e:
        raise Exception(f"Error processing embedding: {e}")


# ============ IMAGE ANIMATION FUNCTIONS ============
def create_blur_frames(image_path: str, image_id: str, num_frames: int = ANIMATION_FRAMES) -> List[str]:
    """
    Create multiple blur levels of an image for animation
    Returns list of base64-encoded images from most blurry to clearest
    """
    if not os.path.exists(image_path):
        return []

    cache_key = f"{image_id}_{num_frames}"
    cache_file = os.path.join(BLUR_CACHE_FOLDER, f"{cache_key}.json")

    # Try to load from cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                frames = json.load(f)
            return frames
        except:
            pass

    # Create blur frames
    try:
        img = Image.open(image_path)
        frames = []
        
        # Calculate blur levels (exponential decay for more natural effect)
        max_blur = 15  # Maximum blur radius
        for i in range(num_frames):
            # Exponential decay: more frames at lower blur levels
            progress = i / (num_frames - 1)
            blur_radius = max_blur * (1 - progress) ** 2
            
            if blur_radius < 0.5:
                blurred = img.copy()
            else:
                blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Convert to base64
            buffer = BytesIO()
            blurred.save(buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            frames.append(f"data:image/jpeg;base64,{img_base64}")
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(frames, f)
        
        return frames
        
    except Exception as e:
        print(f"⚠️ Error creating blur frames: {e}")
        return []


def generate_animation_html(image_paths: List[str], prompts: List[str], scores: List[float]) -> str:
    """
    Generate HTML with diffusion-like animation for all result images
    """
    if not image_paths or not any(image_paths):
        return "<div style='text-align: center; padding: 40px; color: #666;'>No images to display</div>"
    
    # Create blur frames for each image
    all_frames = []
    for i, img_path in enumerate(image_paths):
        if img_path and os.path.exists(img_path):
            frames = create_blur_frames(img_path, f"img_{i}")
            all_frames.append(frames)
        else:
            all_frames.append([])

    # Generate HTML with JavaScript animation
    html = """
    <div style="font-family: Arial, sans-serif; padding: 20px;">
        <style>
            @keyframes denoise {
                0% { filter: blur(15px); opacity: 0.3; transform: scale(1.05); }
                50% { filter: blur(5px); opacity: 0.7; transform: scale(1.0); }
                100% { filter: blur(0px); opacity: 1; transform: scale(1.0); }
            }
            
            .animation-container {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin: 20px 0;
            }
            
            .image-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                padding: 15px;
                box-shadow: 0 8px 16px rgba(0,0,0,0.2);
                transition: transform 0.3s ease;
            } 
            
            .image-card:hover {
                transform: translateY(-5px);
            }
            
            .image-frame {
                width: 100%;
                height: 400px;
                border-radius: 10px;
                overflow: hidden;
                background: #000;
                position: relative;
            }
            
            .image-frame img {
                width: 100%;
                height: 100%;
                object-fit: contain;
                animation: denoise 5s ease-out forwards;
            }
            
            .score-badge {
                background: white;
                color: #667eea;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 16px;
                font-weight: bold;
                display: inline-block;
                margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            
            .prompt-box {
                background: rgba(255,255,255,0.95);
                padding: 15px;
                border-radius: 8px;
                font-size: 15px;
                line-height: 1.6;
                color: #333;
                max-height: 150px;
                overflow-y: auto;
            }
            
            .progress-bar {
                width: 100%;
                height: 4px;
                background: rgba(255,255,255,0.3);
                border-radius: 2px;
                margin-top: 10px;
                overflow: hidden;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                width: 0%;
                animation: progress 5s ease-out forwards;
            }
            
            @keyframes progress {
                0% { width: 0%; }
                100% { width: 100%; }
            }
            
            .loading-overlay {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.7);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 18px;
                z-index: 10;
            }
        </style>
    
        <div class="animation-container">
    """

    for i in range(TOP_K):
        if i < len(image_paths) and image_paths[i] and os.path.exists(image_paths[i]) and all_frames[i]:
            # Use the clearest frame as the final image (JavaScript will animate)
            final_img = all_frames[i][-1]
            
            html += f"""
            <div class="image-card">
                <div class="score-badge">🎯 Match #{i+1} - Similarity: {scores[i]:.4f}</div>
                <div class="image-frame" id="frame_{i}">
                    <div class="loading-overlay" id="loading_{i}">
                        🎨 Denoising... <span id="timer_{i}">0.0</span>s
                    </div>
                    <img src="{final_img}" id="img_{i}" alt="Result {i+1}" />
                </div>
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
                <div class="prompt-box">
                    <strong>📝 Original Prompt (Chinese):</strong><br>
                    {prompts[i][:200]}{'...' if len(prompts[i]) > 200 else ''}
                </div>
                <div style="font-size: 12px; color: rgba(255,255,255,0.8); margin-top: 8px;">
                    📁 {os.path.basename(image_paths[i])}
                </div>
            </div>
            """
        else:
            html += """
            <div class="image-card">
                <div class="score-badge">No Image</div>
                <div class="image-frame" style="display: flex; align-items: center; justify-content: center; background: #333;">
                    <span style="color: #999; font-size: 18px;">Image not available</span>
                </div>
            </div>
            """

    html += """
        </div>
    
        <script>
            // Animation timing
            const duration = 5000; // 5 seconds
            const startTime = Date.now();
            
            function updateTimers() {
                const elapsed = (Date.now() - startTime) / 1000;
                for (let i = 0; i < 5; i++) {
                    const timer = document.getElementById(`timer_${i}`);
                    if (timer) {
                        timer.textContent = Math.min(elapsed, 5.0).toFixed(1);
                    }
                    const loading = document.getElementById(`loading_${i}`);
                    if (loading && elapsed >= 5.0) {
                        loading.style.display = 'none';
                    }
                }
                if (elapsed < 5.0) {
                    requestAnimationFrame(updateTimers);
                }
            }
            
            // Start animation timers
            updateTimers();
        </script>
    </div>
    """

    return html


# ============ FAISS INDEX FUNCTIONS ============
def build_faiss_index(database: List[Dict], progress=gr.Progress()) -> faiss.IndexFlatIP:
    """Build FAISS index from database embeddings"""
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
                    progress((i + 1) / len(database), desc=f"Computing embeddings... ({i+1}/{len(database)})")
                    print(f"  Processed {i + 1}/{len(database)} prompts...")
        except Exception as e:
            print(f"  ⚠️ Error at index {i}: {e}")
            continue

    if not embeddings:
        raise Exception("No valid embeddings computed!")

    embeddings_array = np.array(embeddings, dtype=np.float32)

    print(f"\n✅ Computed {len(embeddings_array)} embeddings")
    print(f"📊 Embedding shape: {embeddings_array.shape}")

    progress(0.8, desc="Building FAISS index...")
    print("\n🔨 Creating FAISS index...")

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings_array)

    print(f"✅ FAISS index built with {index.ntotal} vectors")

    progress(0.9, desc="Saving index to disk...")
    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"💾 FAISS index saved to: {FAISS_INDEX_FILE}")

    metadata = {
        'valid_indices': valid_indices,
        'total_entries': len(database),
        'embedding_dim': EMBEDDING_DIM,
        'database_file': DATABASE_FILE
    }

    with open(FAISS_METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"💾 Metadata saved to: {FAISS_METADATA_FILE}")

    progress(1.0, desc="Index built successfully!")

    return index


def load_faiss_index() -> Tuple[faiss.IndexFlatIP, Dict]:
    """Load FAISS index and metadata from disk"""
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_FILE}\nPlease build the index first.")
    if not os.path.exists(FAISS_METADATA_FILE):
        raise FileNotFoundError(f"FAISS metadata not found: {FAISS_METADATA_FILE}\nPlease build the index first.")

    index = faiss.read_index(FAISS_INDEX_FILE)
    print(f"✅ Loaded FAISS index with {index.ntotal} vectors")

    with open(FAISS_METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"✅ Loaded metadata: {metadata['total_entries']} total entries, {len(metadata['valid_indices'])} valid")

    return index, metadata


def load_database(db_file: str) -> List[Dict]:
    """Load the JSON database"""
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"Database file not found: {db_file}")
    with open(db_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✅ Loaded {len(data)} entries from database")
    return data


# ============ SEARCH FUNCTIONS ============
def search_with_faiss(
    query: str,
    index: faiss.IndexFlatIP,
    metadata: Dict,
    database: List[Dict],
    top_k: int = 5
) -> Tuple[List[str], List[str], List[float]]:
    """Search for similar images using FAISS"""
    if not query.strip():
        raise ValueError("Query cannot be empty")
    print(f"\n🔍 Searching for: '{query}'")
    query_embedding = get_embedding(query)
    query_embedding = query_embedding.reshape(1, -1)

    scores, indices = index.search(query_embedding, top_k)

    image_paths = []
    prompts = []
    final_scores = []

    valid_indices = metadata['valid_indices']

    for i, (score, faiss_idx) in enumerate(zip(scores[0], indices[0])):
        if faiss_idx == -1:
            continue
        
        db_idx = valid_indices[faiss_idx]
        entry = database[db_idx]
        
        filename = entry.get('filename', '')
        prompt = entry.get('prompt', 'No prompt available')
        
        if filename:
            if os.path.isabs(filename):
                img_path = filename
            else:
                img_path = os.path.join(IMAGES_FOLDER, os.path.basename(filename))
        else:
            img_path = None
        
        image_paths.append(img_path)
        prompts.append(prompt)
        final_scores.append(float(score))
        
        print(f"  Match {i+1}: score={score:.4f}, db_idx={db_idx}, prompt='{prompt[:50]}...'")

    return image_paths, prompts, final_scores


# ============ GRADIO INTERFACE ============
def build_index_ui(progress=gr.Progress()) -> str:
    """UI function to build FAISS index"""
    try:
        database = load_database(DATABASE_FILE)
        index = build_faiss_index(database, progress)
        return f"✅ FAISS index built successfully with {index.ntotal} vectors!"
    except Exception as e:
        return f"❌ Error building index: {str(e)}"


def perform_search(query: str, progress=gr.Progress()) -> Tuple[Image.Image, List, str, str]:
    """Main search function with diffusion animation"""
    if not query.strip():
        best_image = None
        empty_gallery = [None] * 4  # Now showing 4 remaining images
        empty_details = "❌ Please enter a search query"
        empty_animation = "<div style='text-align: center; padding: 40px; color: #666;'>Enter a search query to see results with diffusion animation</div>"
        return best_image, empty_gallery, empty_details, empty_animation
    
    try:
        progress(0, desc="Loading FIG model...")
        
        database = load_database(DATABASE_FILE)
        index, metadata = load_faiss_index()
        
        progress(0.3, desc="Generating Images...")
        image_paths, prompts, scores = search_with_faiss(
            query, index, metadata, database, TOP_K
        )
        
        # Best match (index 0) - displayed separately
        best_image = None
        best_details_html = ""
        
        if len(image_paths) > 0 and image_paths[0] and os.path.exists(image_paths[0]):
            try:
                best_image = Image.open(image_paths[0])
                best_details_html = f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           padding: 20px; border-radius: 15px; margin: 10px 0; 
                           box-shadow: 0 6px 12px rgba(0,0,0,0.2);">
                    <div style="background: white; padding: 15px; border-radius: 10px; 
                               margin-bottom: 15px; display: inline-block; 
                               font-size: 22px; font-weight: bold; color: #667eea; 
                               border: 3px solid #667eea;">
                        🏆 BEST MATCH - Similarity: {scores[0]:.4f}
                    </div>
                    <div style="background: rgba(255,255,255,0.95); padding: 20px; 
                               border-radius: 10px; font-size: 18px; line-height: 1.8; 
                               color: #333;">
                        <strong style="color: #667eea; font-size: 20px;">📝 Original Prompt (Chinese):</strong><br>
                        {prompts[0]}
                    </div>
                    <div style="margin-top: 15px; font-size: 16px; color: rgba(255,255,255,0.9); 
                               background: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px;">
                        📁 Image: {os.path.basename(image_paths[0])}
                    </div>
                </div>
                """
            except Exception as e:
                print(f"⚠️ Error loading best image {image_paths[0]}: {e}")
                best_details_html = f"<div style='color: red; padding: 10px;'>Error loading best image: {e}</div>"
        
        # Remaining matches (indices 1-4) - displayed in gallery
        gallery_results = []
        detailed_html = []
        
        for i in range(1, TOP_K):  # Start from index 1, not 0
            if i < len(image_paths) and image_paths[i] and os.path.exists(image_paths[i]):
                try:
                    img = Image.open(image_paths[i])
                    gallery_label = f"#{i+1} - Score: {scores[i]:.4f}"
                    gallery_results.append((img, gallery_label))
                    
                    prompt_text = prompts[i]
                    if len(prompt_text) > 300:
                        prompt_display = prompt_text[:300] + "..."
                    else:
                        prompt_display = prompt_text
                    
                    detailed_html.append(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                               padding: 15px; border-radius: 10px; margin: 10px 0; 
                               box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <div style="background: white; padding: 10px; border-radius: 5px; 
                                   margin-bottom: 10px; display: inline-block; 
                                   font-size: 16px; font-weight: bold; color: #667eea;">
                            🎯 Match #{i+1} - Similarity: {scores[i]:.4f}
                        </div>
                        <div style="background: rgba(255,255,255,0.95); padding: 15px; 
                                   border-radius: 5px; font-size: 14px; line-height: 1.6; 
                                   color: #333;">
                            <strong style="color: #667eea; font-size: 16px;">📝 Original Prompt (Chinese):</strong><br>
                            {prompt_display}
                        </div>
                        <div style="margin-top: 10px; font-size: 12px; color: rgba(255,255,255,0.9);">
                            📁 Image: {os.path.basename(image_paths[i])}
                        </div>
                    </div>
                    """)
                except Exception as e:
                    print(f"⚠️ Error loading image {image_paths[i]}: {e}")
                    gallery_results.append((None, f"Error: {e}"))
                    detailed_html.append(f"<div style='color: red; padding: 10px;'>Error loading image: {e}</div>")
            else:
                gallery_results.append((None, "Image not found"))
                detailed_html.append("<div style='padding: 10px;'>Image not found or invalid path</div>")
        
        detailed_html_str = "<div style='font-family: Arial, sans-serif;'>" + \
                           (best_details_html if best_details_html else "") + \
                           "<h3 style='color: white; margin-top: 30px;'>📋 Other Matches:</h3>" + \
                           "".join(detailed_html) + "</div>"
        
        # Generate diffusion animation HTML for all 5 results
        animation_html = generate_animation_html(image_paths, prompts, scores)
        
        progress(1.0, desc="Done!")
        return best_image, gallery_results, detailed_html_str, animation_html
    
    except FileNotFoundError as e:
        error_msg = f"⚠️ {str(e)}\n\nPlease build the FAISS index first using the button below."
        empty_gallery = [None] * 4
        empty_animation = "<div style='text-align: center; padding: 40px; color: #666;'>⚠️ Please build the FAISS index first</div>"
        return None, empty_gallery, error_msg, empty_animation
    except Exception as e:
        print(f"❌ Search error: {e}")
        import traceback
        traceback.print_exc()
        error_msg = f"❌ Error: {str(e)}"
        empty_gallery = [None] * 4
        empty_animation = f"<div style='text-align: center; padding: 40px; color: red;'>❌ Error: {str(e)}</div>"
        return None, empty_gallery, error_msg, empty_animation


def check_index_status() -> str:
    """Check if FAISS index exists and return status"""
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_METADATA_FILE):
        try:
            index = faiss.read_index(FAISS_INDEX_FILE)
            with open(FAISS_METADATA_FILE, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return f"✅ FAISS index ready\n• Vectors: {index.ntotal:,}\n• Database: {metadata['total_entries']:,} entries\n• File: {FAISS_INDEX_FILE}"
        except Exception as e:
            return f"⚠️ Index exists but corrupted: {e}\nPlease rebuild."
    else:
        return "❌ FAISS index not found\nPlease build the index first."


def create_interface():
    """Create the Gradio interface"""
    custom_css = """
.search-box textarea {
    font-size: 18px;
    font-weight: 500;
}
.gallery-item {
    text-align: center;
}
.gallery-item img {
    max-height: 400px;
    object-fit: contain;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.gallery-label {
    font-size: 16px;
    font-weight: bold;
    color: #667eea;
    margin-top: 10px;
}
.results-container {
    background: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    margin: 20px 0;
}
.animation-section {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 30px;
    border-radius: 15px;
    margin: 20px 0;
    box-shadow: 0 8px 16px rgba(0,0,0,0.3);
}
.best-match-container {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 30px;
    border-radius: 15px;
    margin: 20px 0;
    box-shadow: 0 8px 16px rgba(0,0,0,0.3);
}
.best-match-image {
    border: 4px solid #667eea;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5);
}
"""

    with gr.Blocks(
        title="Multilingual Image Fake Generation",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        gr.Markdown(
            """
            # 🎨 Multilingual Semantic Image Search with Diffusion Animation
            
            Search for images using natural language queries in **English**. 
            Results are displayed with a **diffusion-like denoising animation** (5 seconds).
            
            **Model:** bge-m3-Q8_0-GGUF (Multilingual) | **Database:** kaupane/nano-banana-pro-gen | **Results:** Top 5 matches
            """
        )
        
        # Index Status
        with gr.Row():
            status_output = gr.Textbox(
                label="📊 FAISS Index Status",
                value=check_index_status(),
                interactive=False,
            )
        
        # Build Index Section
        with gr.Accordion("🔨 Build/Rebuild FAISS Index", open=False):
            gr.Markdown(
                """
                **First time setup:** Click the button below to build the FAISS index.
                This will compute embeddings for all prompts and save them for fast search.
            """
            )
            build_btn = gr.Button("🔨 Build FAISS Index", variant="secondary")
            build_output = gr.Textbox(label="Build Status", interactive=False)
            
            build_btn.click(
                fn=build_index_ui,
                inputs=[],
                outputs=[build_output]
            )
        
        # Search Section 
        gr.Markdown("### 🔍 Search/Generate Images")
        
        with gr.Row():
            query_input = gr.Textbox(
                label="🖼️ Generate Images (English)",
                placeholder="e.g., 'space station with solar panels', 'tank in battlefield', 'ship in fog'...",
                lines=2,
                elem_classes=["search-box"],
                scale=4
            )
        
        search_btn = gr.Button(" Generate Images", variant="primary", size="lg", scale=1)
        
        # Best Match Section - Displayed Prominently
        gr.Markdown("### 🏆 Best Match")
        gr.Markdown("*The highest scoring result*")
        
        with gr.Row(elem_classes=["best-match-container"]):
            best_match_image = gr.Image(
                label="🥇 Best Match",
                show_label=True,
                height=500,
                elem_classes=["best-match-image"]
            )
        
        # Other Matches Gallery
        gr.Markdown("### 📸 Other Matches (Top 2-5)")
        gr.Markdown("*Additional similar results*")
        
        with gr.Row():
            results_gallery = gr.Gallery(
                label="Matches #2-5",
                show_label=True,
                columns=2,
                rows=2,
                object_fit="contain",
                height="auto",
                allow_preview=True,
                preview=True,
                scale=1,
                selected_index=0
            )
        
        # Detailed Results with Large Text
        gr.Markdown("### 📋 Detailed Information")
        detailed_results_html = gr.HTML(
            label="Detailed Information",
            show_label=True,
            elem_classes=["results-container"]
        )
        
        # Diffusion Animation Section
        with gr.Accordion(visible=False):
            gr.Markdown("### 🎬 Diffusion Denoising Animation")
            gr.Markdown("*Watch images gradually denoise from blurry to clear (5 seconds)*")
            
            diffusion_animation = gr.HTML(
                label="Diffusion Animation",
                show_label=True,
                elem_classes=["animation-section"]
            )
        
        # Examples
        gr.Markdown("### 💡 Example Queries")
        gr.Examples(
            examples=[
                "space station with large solar panels in deep space",
                "military tank driving on open field",
                "naval ship sailing through fog and snow",
                "impressionist landscape painting with snow",
                "fighter jet flying in the sky",
                "ancient Chinese architecture with red walls"
            ],
            inputs=query_input,
            label="Click to try:"
        )
        
        # Instructions
        with gr.Accordion("ℹ️ How it works", open=False):
            gr.Markdown(
                """
                ## Workflow
                
                ### First Time Setup:
                1. Click "Build FAISS Index" (takes several minutes)
                2. Embeddings are computed and saved to disk
                3. Index is ready for fast search
                
                ### Search:
                1. Enter your query in English
                2. Query is converted to embedding using bge-m3
                3. FAISS performs fast similarity search
                4. Results displayed in 3 formats:
                   - **Best Match**: Prominently displayed (#1 result)
                   - **Gallery**: Other matches (#2-5)
                   - **Detailed Info**: Full prompts and scores
                   - **Diffusion Animation**: Blurry-to-clear animation (5s)
                
                **Animation Effect:**
                - Images start heavily blurred (simulating noise)
                - Gradually become clearer over 5 seconds
                - Mimics diffusion model generation process
                - Progress bar shows animation timing
                """
            )
        
        # Event handlers
        search_btn.click(
            fn=perform_search,
            inputs=[query_input],
            outputs=[best_match_image, results_gallery, detailed_results_html, diffusion_animation]
        )
        
        query_input.submit(
            fn=perform_search,
            inputs=[query_input],
            outputs=[best_match_image, results_gallery, detailed_results_html, diffusion_animation]
        )
        
        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
        refresh_btn.click(
            fn=check_index_status,
            inputs=[],
            outputs=[status_output]
        )
        
        gr.Markdown(
            """
            ---
            **Powered by:** llama.cpp + bge-m3 + FAISS + Gradio + Diffusion Animation
            """
        )

    return demo


# ============ MAIN ============
if __name__ == "__main__":
    print("="*70)
    print("🚀 Starting Multilingual Image Search App with Diffusion Animation")
    print("="*70)
    print(f"📊 Database: {DATABASE_FILE}")
    print(f"📁 Images folder: {IMAGES_FOLDER}")
    print(f"🌐 llama.cpp server: {LLAMA_CPP_SERVER_URL}")
    print(f"🤖 Model: bge-m3-Q8_0-GGUF")
    print(f"🔍 FAISS index: {FAISS_INDEX_FILE}")
    print(f"🎬 Animation: {ANIMATION_DURATION} seconds, {ANIMATION_FRAMES} frames")
    print(f"📈 Top-K Results: {TOP_K}")
    print("="*70)
    print("\n🔌 Testing connection to llama.cpp server...")
    try:
        response = requests.get(f"{LLAMA_CPP_SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Connected to llama.cpp server")
        else:
            print(f"⚠️ Server responded with status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to llama.cpp server!")
        print(f"   Please start the server with: ")
        print(f"   ./llama-server -m models/bge-m3-Q8_0.gguf --port 8080 --embedding")
        exit(1)
    except Exception as e:
        print(f"⚠️ Connection test failed: {e}")

    print("\n" + "="*70)
    if os.path.exists(FAISS_INDEX_FILE):
        try:
            index = faiss.read_index(FAISS_INDEX_FILE)
            print(f"✅ FAISS index found with {index.ntotal:,} vectors")
            print(f"   Ready for instant search!")
        except Exception as e:
            print(f"⚠️ FAISS index exists but may be corrupted: {e}")
            print(f"   Please rebuild the index from the UI.")
    else:
        print(f"⚠️ FAISS index not found: {FAISS_INDEX_FILE}")
        print(f"   Please build the index from the UI first.")
    print("="*70 + "\n")

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )