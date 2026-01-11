#!/usr/bin/env python3
"""
webptopdf_multi.py
1. Accepts multiple input folders.
2. Finds archive name from the first folder's 'sibling'.
3. Merges all images into one PDF in HOME/translated/.
4. Renames input folders to the archive name.
"""

import re
import sys
import os
from pathlib import Path
from PIL import Image

def extract_number(filename):
    """Extract first number from filename for smart sorting."""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

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
    return sorted(list(set(files)), key=lambda p: extract_number(p.stem))

def webp_multi_to_pdf(folder_list, output_pdf):
    all_images = []
    total_pages = 0
    
    print("Collecting images from folders...")
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
        total_pages += len(files)

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

def main():
    if len(sys.argv) < 2:
        print("Usage: python a.py <folder1-t> [folder2-t ...]")
        sys.exit(1)

    folders = [Path(f).resolve() for f in sys.argv[1:]]
    
    # --- FIND ARCHIVE NAME ---
    # Look at the first folder provided (e.g., manga1-t)
    first_folder = folders[0]
    home_dir = first_folder.parent
    
    # Target the sibling folder (manga1-t -> manga1)
    sibling_name = first_folder.name.replace("-translated", "")
    sibling_folder = home_dir / sibling_name
    
    archive_name = None
    if sibling_folder.exists() and sibling_folder.is_dir():
        for item in sibling_folder.iterdir():
            if item.suffix.lower() in {'.zip', '.7z', '.rar'}:
                archive_name = item.stem
                break
    
    if not archive_name:
        print(f"Could not find archive in {sibling_folder}. Using fallback.")
        archive_name = sibling_name

    # --- DEFINE OUTPUT ---
    pdf_output = home_dir / "translated" / f"{archive_name}.pdf"

    # --- EXECUTE MERGE ---
    success = webp_multi_to_pdf(folders, pdf_output)

    # --- RENAME FOLDERS ---
    if success:
        print("\nRenaming folders...")
        for i, folder_path in enumerate(folders):
            # If multiple folders, add suffix (abc_01, abc_02)
            suffix = f"_{i+1:02d}" if len(folders) > 1 else ""
            new_name = f"{archive_name}{suffix}"
            new_path = folder_path.parent / new_name
            
            if folder_path.exists() and not new_path.exists():
                try:
                    folder_path.rename(new_path)
                    print(f"  Renamed: {folder_path.name} -> {new_name}")
                except Exception as e:
                    print(f"  Error renaming {folder_path.name}: {e}")
            else:
                print(f"  Skip rename: {new_path.name} already exists.")
        
        print("\nDone! All images merged and folders renamed.")

if __name__ == "__main__":
    main()