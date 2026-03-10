# pip install datasets requests tqdm huggingface_hub pandas pyarrow
# yu give a dataset repo hf
# you download all the images
# you get a staring JSON database before preparing the FAISS one
from datasets import load_dataset
import pandas as pd
from PIL import Image
import io
import json
from time import sleep
import os
import requests
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
from pathlib import Path

# ============ HELPER FUNCTIONS ============

def list_parquet_files(repo_name):
    """List all parquet files in the dataset repo"""
    print(f"\n🔍 Searching for parquet files in {repo_name}...")
    
    try:
        api = HfApi()
        files = api.list_repo_files(repo_name, repo_type="dataset")
        parquet_files = [f for f in files if f.endswith('.parquet')]
        
        if not parquet_files:
            print("❌ No parquet files found in this dataset!")
            return []
        
        print(f"✅ Found {len(parquet_files)} parquet file(s):")
        for i, pf in enumerate(parquet_files, 1):
            print(f"  {i}. {pf}")
        
        return parquet_files
    
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        return []


def download_parquet_files(repo_name, parquet_files, num_files, download_folder="./parquet_cache"):
    """Download selected parquet files from the repo"""
    os.makedirs(download_folder, exist_ok=True)
    
    # Ask user how many files to download
    if num_files is None:
        num_files_input = input(f"\n📥 How many parquet files to download? (1-{len(parquet_files)}, default all): ").strip()
        if num_files_input:
            num_files = min(int(num_files_input), len(parquet_files))
        else:
            num_files = len(parquet_files)
    
    # Select files (first N files)
    selected_files = parquet_files[:num_files]
    
    print(f"\n📥 Downloading {len(selected_files)} parquet file(s)...")
    
    downloaded_paths = []
    for pf in tqdm(selected_files, desc="📥 Downloading parquet", unit="files", ncols=100):
        try:
            local_path = hf_hub_download(
                repo_id=repo_name,
                filename=pf,
                repo_type="dataset",
                cache_dir=download_folder
            )
            downloaded_paths.append(local_path)
        except Exception as e:
            print(f"\n❌ Error downloading {pf}: {e}")
            continue
    
    return downloaded_paths


def inspect_parquet_files(parquet_paths):
    """Inspect parquet files and return metadata"""
    print("\n" + "="*70)
    print("📊 PARQUET FILES INSPECTION")
    print("="*70)
    
    total_rows = 0
    all_metadata = []
    
    for i, pq_path in enumerate(parquet_paths, 1):
        try:
            # Get file size
            file_size = os.path.getsize(pq_path) / (1024 * 1024)  # MB
            
            # Read parquet metadata
            parquet_file = pq.ParquetFile(pq_path)
            num_rows = parquet_file.metadata.num_rows
            num_columns = parquet_file.metadata.num_columns
            schema = parquet_file.schema_arrow
            
            total_rows += num_rows
            
            print(f"\n📄 File {i}/{len(parquet_paths)}: {os.path.basename(pq_path)}")
            print(f"  • Size: {file_size:.2f} MB")
            print(f"  • Rows: {num_rows:,}")
            print(f"  • Columns: {num_columns}")
            print(f"  • Schema:")
            
            # Show schema (first 10 fields)
            for j, field in enumerate(schema):
                if j < 10:
                    print(f"      - {field.name}: {field.type}")
                elif j == 10:
                    print(f"      ... and {len(schema) - 10} more fields")
                    break
            
            # Read first row for preview
            df_sample = pd.read_parquet(pq_path, engine='pyarrow').head(1)
            print(f"\n  • First row preview:")
            for col in df_sample.columns:
                val = df_sample[col].iloc[0]
                val_str = str(val)
                if len(val_str) > 80:
                    val_str = val_str[:80] + "..."
                print(f"      {col}: {val_str}")
            
            all_metadata.append({
                'path': pq_path,
                'filename': os.path.basename(pq_path),
                'size_mb': file_size,
                'num_rows': num_rows,
                'num_columns': num_columns
            })
            
        except Exception as e:
            print(f"\n❌ Error inspecting {pq_path}: {e}")
            continue
    
    print("\n" + "="*70)
    print(f"📈 TOTAL: {len(parquet_paths)} file(s), {total_rows:,} rows")
    print("="*70 + "\n")
    
    return all_metadata, total_rows


def get_existing_last_id(json_filename):
    """Get the last ID from existing JSON file to continue incremental numbering"""
    if os.path.exists(json_filename):
        try:
            with open(json_filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if existing_data and len(existing_data) > 0:
                    last_id = existing_data[-1].get('id', 0)
                    print(f"📁 Found existing file: {json_filename}")
                    print(f"📊 Existing entries: {len(existing_data)}")
                    print(f"🔢 Last ID: {last_id}")
                    return last_id + 1, existing_data
        except Exception as e:
            print(f"⚠️ Could not read existing file: {e}")
    return 0, []


def save_progress(json_filename, dumpfile):
    """Save dumpfile to JSON"""
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(dumpfile, f, indent=2, ensure_ascii=False)


# ============ MAIN SCRIPT ============

print("\n" + "🎯"*35)
print("   HF DATASET OFFLINE DOWNLOADER")
print("🎯"*35 + "\n")

# Step 1: Get repo name
repo = input("Paste the HF dataset repo name (e.g., 'cerealt/open-image-preferences-v1-binarized'): ").strip()
if not repo:
    print("❌ Repo name is required!")
    exit(1)

# Step 2: List parquet files
parquet_files = list_parquet_files(repo)
if not parquet_files:
    print("❌ No parquet files found. Exiting.")
    exit(1)

# Step 3: Download parquet files
download_folder = input("\n📁 Parquet cache folder (default './parquet_cache'): ").strip()
if not download_folder:
    download_folder = "./parquet_cache"

downloaded_paths = download_parquet_files(repo, parquet_files, num_files=None, download_folder=download_folder)

if not downloaded_paths:
    print("❌ No files downloaded. Exiting.")
    exit(1)

# Step 4: Inspect parquet files
metadata, total_rows = inspect_parquet_files(downloaded_paths)

# Step 5: Configuration
print("\n⚙️  DOWNLOAD CONFIGURATION")
print("-"*70)

image_folder = input("Enter destination folder for images (e.g., './images'): ").strip()
if not image_folder:
    image_folder = "./images"

os.makedirs(image_folder, exist_ok=True)

# Generate JSON filename from repo name
json_filename = f"{repo.replace('/', '_')}.json"

# Step 6: Get download range
print(f"\n📍 DOWNLOAD RANGE:")
print(f"💡 Total available rows across all downloaded parquet files: {total_rows:,}")
print(f"   Choose a range within 0 to {total_rows-1:,}")

start_input = input(f"Start index (default 0): ").strip()
start_index = int(start_input) if start_input else 0

if start_index >= total_rows:
    print(f"❌ Start index {start_index} exceeds total rows ({total_rows})")
    exit(1)

end_input = input(f"End index (exclusive, default: start+50): ").strip()
if end_input:
    end_index = int(end_input)
    if end_index <= start_index:
        print("❌ End index must be greater than start index!")
        exit(1)
    if end_index > total_rows:
        print(f"⚠️ End index {end_index} exceeds total rows ({total_rows}). Clamping to {total_rows}")
        end_index = total_rows
else:
    end_index = None

if end_index:
    num_sample = end_index - start_index
else:
    num_sample_input = input(f"No end index specified. How many samples to download? (default 50): ").strip()
    num_sample = int(num_sample_input) if num_sample_input else 50

# Ask about displaying images
display_images = input("\n🖼️ Display images after download? (y/n, default n): ").strip().lower()
show_images = display_images in ['y', 'yes', 'true', '1']

sleep_time = 6
if show_images:
    delay_input = input(f"Delay between images in seconds (default {sleep_time}): ").strip()
    if delay_input:
        sleep_time = int(delay_input)

print(f"\n📋 CONFIGURATION SUMMARY:")
print(f"  Repo: {repo}")
print(f"  Parquet files: {len(downloaded_paths)}")
print(f"  Total rows available: {total_rows:,}")
print(f"  Image folder: {image_folder}")
print(f"  JSON file: {json_filename}")
print(f"  Range: {start_index} to {end_index if end_index else start_index + num_sample} (exclusive)")
print(f"  Display images: {'Yes' if show_images else 'No'}")
if show_images:
    print(f"  Delay: {sleep_time} seconds")
print("="*70 + "\n")

# Check for existing JSON file and get last ID
start_id, existing_data = get_existing_last_id(json_filename)
dumpfile = existing_data

confirm = input("Proceed with download? (y/n): ").strip().lower()
if confirm not in ['y', 'yes']:
    print("❌ Download cancelled.")
    exit(0)

# ============ LOAD DATASET FROM PARQUET ============
print(f"\n🚀 Loading dataset from {len(downloaded_paths)} parquet file(s)...")

try:
    # Load dataset from local parquet files
    dataset = load_dataset(
        'parquet', 
        data_files=downloaded_paths, 
        split='train',
        streaming=True
    )
    
    print(f"✅ Dataset loaded successfully!")
    print(f"🔢 Starting ID: {start_id}")
    print("-"*70 + "\n")
    
    # Skip to start index
    if start_index > 0:
        print(f"⏩ Skipping first {start_index} samples...")
        for _ in tqdm(range(start_index), desc="⏩ Skipping", unit="samples", ncols=100):
            next(iter(dataset))
    
    # Determine end point
    if end_index:
        total_to_download = end_index - start_index
    else:
        total_to_download = num_sample
    
    im_id = start_id
    downloaded = 0
    failed = 0
    mydelta = 2000
    
    # Create tqdm progress bar
    with tqdm(total=total_to_download, 
              desc="📥 Downloading", 
              unit="images",
              ncols=100,
              colour='green',
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        for i, example in enumerate(dataset):
            if downloaded >= total_to_download:
                break
            
            current_index = start_index + downloaded
            filename = f'{current_index+mydelta}.png'
            full_path = os.path.join(image_folder, filename)
            
            try:
                # Save image for new dataset cerealt/open-image-preferences-v1-binarized
                ########################################################################
                if 'chosen' in example and example['chosen']['bytes'] is not None:
                    image_data1 = example['chosen']['bytes']
                    image1 = Image.open(io.BytesIO(image_data1))
                    image1.save(full_path)
                    
                    # Get prompt
                    prompt = example.get('prompt', '')
                    if not prompt and 'json' in example and isinstance(example['json'], dict):
                        prompt = example['json'].get('prompt') or example['json'].get('caption') or str(example['json'])
                    
                    # Display image if requested
                    if show_images:
                        try:
                            myimage = Image.open(full_path)
                            print(f"\n🖼️ Showing {full_path} (#{current_index})...")
                            if prompt:
                                preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
                                print(f"📝 Prompt: {preview}")
                            myimage.show()
                            print(f"⏳ Waiting {sleep_time} seconds...")
                            sleep(sleep_time)
                        except Exception as e:
                            print(f"⚠️ Could not display image: {e}")
                    
                    # Add to dumpfile
                    dumpfile.append({
                        "id": im_id+mydelta,
                        "index": current_index+mydelta,
                        "prompt": prompt,
                        "filename": full_path
                    })
                    
                    im_id += 1
                    downloaded += 1
                    
                    # Update tqdm progress bar
                    pbar.set_postfix({
                        'ID': f'{start_id}-{im_id-1}',
                        'Failed': failed,
                        'Index': current_index
                    })
                    pbar.update(1)
                    
                    # Save progress periodically
                    if downloaded % 10 == 0:
                        save_progress(json_filename, dumpfile)
                else:
                    print(f"\n⚠️ Sample {current_index}: No image found, skipping...")
                    failed += 1
                    pbar.set_postfix({'Failed': failed})
                    pbar.update(1)
                    
            except Exception as e:
                print(f"\n❌ Error processing sample {current_index}: {e}")
                failed += 1
                pbar.set_postfix({'Failed': failed})
                pbar.update(1)
                continue
    
    # Final save
    save_progress(json_filename, dumpfile)
    
    print("\n" + "="*70)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*70)
    print(f"📊 Successfully downloaded: {downloaded} images")
    print(f"⚠️ Failed/Skipped: {failed}")
    print(f"🔢 ID range: {start_id} to {im_id-1}")
    print(f"📁 Images saved to: {os.path.abspath(image_folder)}")
    print(f"📄 Metadata file: {os.path.abspath(json_filename)}")
    print(f"📈 Total entries in JSON: {len(dumpfile)}")
    print(f"💾 Parquet cache: {os.path.abspath(download_folder)}")
    print("="*70 + "\n")
    
except KeyboardInterrupt:
    print("\n\n⚠️ Download interrupted by user!")
    save_progress(json_filename, dumpfile)
    print(f"✅ Progress saved. {len(dumpfile)} entries in total.")
    
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    save_progress(json_filename, dumpfile)

    print("✅ Partial progress saved.")
