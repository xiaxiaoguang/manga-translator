#!/usr/bin/env python3
"""
webptopdf_multi.py
1. Accepts multiple input folders.
2. Finds archive name from the first folder's 'sibling'.
3. Merges all images into one PDF in HOME/translated/.
4. COPIES and RENAMES all image files into a single consolidated folder 
   using pure sequential numbering (0001, 0002, etc.).
   Original folders are preserved.
"""


import re
import sys
import os
import shutil
from pathlib import Path
from PIL import Image

def get_sorted_image_files(folder_path):
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Warning: Folder not found: {folder}")
        return []
    exts = ("*.webp", "*.jpg", "*.jpeg", "*.png")
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
        files.extend(folder.glob(ext.upper()))
    return sorted(list(set(files)), key=lambda p: (p.name.lower()))

def webp_multi_to_pdf(folder_list, output_pdf):
    all_images = []
    
    print("Collecting images from folders for PDF...")
    for idx, folder_path in enumerate(folder_list):
        files = get_sorted_image_files(folder_path)
        if not files:
            print(f"  Folder {idx+1}: {folder_path} -> No images found")
            continue
        
        print(f"  Folder {idx+1}: {folder_path} -> {len(files)} files")
        for file_path in files:
            try:
                img = Image.open(file_path)
                if img.mode in ("RGBA", "LA", "P"):
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P": img = img.convert("RGBA")
                    mask = img.split()[-1] if "A" in img.getbands() else None
                    bg.paste(img, mask=mask)
                    img = bg
                else:
                    img = img.convert("RGB")
                all_images.append(img)
            except Exception as e:
                print(f"    Failed to load {file_path.name}: {e}")

    if not all_images:
        return False

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving {len(all_images)} pages -> {output_pdf}")
    all_images[0].save(
        output_pdf,
        save_all=True,
        append_images=all_images[1:],
        format="PDF",
        quality=95,
        dpi=(300, 300)
    )
    return True

def merge_and_rename_files(source_folders, target_folder_name):
    """
    Copies files from source_folders into a single target_folder.
    Renames them sequentially (0001, 0002, ...) based on input order.
    Uses get_sorted_image_files to ensure the numerical sequence is correct.
    """
    if not source_folders:
        return

    base_dir = source_folders[0].parent
    target_path = base_dir / target_folder_name
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying and sequentially renaming files into: {target_path.name}/")

    global_counter = 1
    
    for folder in source_folders:
        if not folder.exists():
            continue
            
        # IMPORTANT: Use get_sorted_image_files here too to maintain the 
        files = get_sorted_image_files(folder)
        
        for item in files:
            # Create new name like 0001.jpg, 0002.webp, etc.
            new_name = f"{global_counter:04d}{item.suffix.lower()}"
            dest_file = target_path / new_name
            
            try:
                # Use shutil.copy2 to preserve metadata and keep original files
                shutil.copy2(str(item), str(dest_file))
                global_counter += 1
            except Exception as e:
                print(f"  Error copying/renaming {item.name} to {new_name}: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python webptopdf_multi.py <folder1-t> [folder2-t ...]")
        sys.exit(1)

    folders = [Path(f).resolve() for f in sys.argv[1:]]
    
    # --- FIND ARCHIVE NAME ---
    first_folder = folders[0]
    home_dir = first_folder.parent
    
    # Target the sibling folder (manga1-t -> manga1)
    sibling_name = first_folder.name.replace("-translated", "").replace("-t", "")
    sibling_folder = home_dir / sibling_name
    
    archive_name = None
    if sibling_folder.exists() and sibling_folder.is_dir():
        for item in sibling_folder.iterdir():
            if item.suffix.lower() in {'.zip', '.7z', '.rar'}:
                archive_name = item.stem
                break
    
    if not archive_name:
        print(f"Could not find archive in {sibling_folder}. Using folder name as fallback.")
        archive_name = sibling_name

    # --- DEFINE OUTPUT PDF ---
    pdf_output = home_dir / "translated" / f"{archive_name}.pdf"

    # --- EXECUTE PDF MERGE ---
    success = webp_multi_to_pdf(folders, pdf_output)

    # --- EXECUTE FILE MERGE & RENAME ---
    if success:
        merge_and_rename_files(folders, archive_name)
        print(f"\nDone! PDF created and images copied to '{archive_name}' with sequential names.")
        print("Original folders have been preserved.")
    else:
        print("\nProcess failed: No images were found or processed.")

if __name__ == "__main__":
    main()