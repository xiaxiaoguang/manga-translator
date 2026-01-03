#!/usr/bin/env python3
"""
webptopdf_multi.py
Merge WebP images from MULTIPLE folders in given order into ONE PDF.
Automatically renumbers pages sequentially (0001, 0002, ...) across folders.
Smart sorting inside each folder.
"""

import re
import sys
from pathlib import Path
from PIL import Image

def extract_number(filename):
    """Extract first number from filename, return large number if none"""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def get_sorted_webp_files(folder_path):
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Warning: Folder not found: {folder}")
        return []
    files = list(folder.glob("*.webp")) + list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    return sorted(files, key=lambda p: extract_number(p.stem))

def webp_multi_to_pdf(folder_list, output_pdf=None):
    all_images = []
    total_pages = 0

    print("Collecting and sorting WebP files from folders...\n")

    for idx, folder_path in enumerate(folder_list):
        files = get_sorted_webp_files(folder_path)
        if not files:
            print(f"  Folder {idx+1}: {folder_path} → No WebP files")
            continue

        print(f"  Folder {idx+1}: {folder_path} → {len(files)} files")
        for i, f in enumerate(files, 1):
            num = extract_number(f.stem)
            print(f"    → Page {total_pages + i:04d} : {f.name} (orig #{num if num != float('inf') else '?'})")

        for file_path in files:
            try:
                img = Image.open(file_path)
                if img.mode in ("RGBA", "LA", "P"):
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    bg.paste(img, mask=img.split()[-1] if "A" in img.getbands() else None)
                    img = bg
                else:
                    img = img.convert("RGB")
                all_images.append(img)
            except Exception as e:
                print(f"    Failed to load {file_path.name}: {e}")

        total_pages += len(files)

    if not all_images:
        print("No images loaded. Exiting.")
        return

    # Output PDF
    if output_pdf is None:
        parent = Path(folder_list[0]).parent
        output_pdf = parent / f"COMBINED_{len(folder_list)}_chapters.pdf"
    else:
        output_pdf = Path(output_pdf)

    print(f"\nSaving {len(all_images)} pages → {output_pdf}")
    all_images[0].save(
        output_pdf,
        save_all=True,
        append_images=all_images[1:],
        format="PDF",
        quality=95,
        dpi=(300, 300)
    )
    print("Done! PDF created successfully!")

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python webptopdf_multi.py <folder1> <folder2> [folder3 ...] [output.pdf]")
        print("\nExample:")
        print("  python webptopdf_multi.py l1 l2 l3 l4 \"Volume 1.pdf\"")
        print("  python webptopdf_multi.py ch001 ch002 ch003")
        sys.exit(1)

    folders = []
    output = None

    for arg in sys.argv[1:]:
        if arg.endswith(('.pdf', '.PDF')) and Path(arg).suffix.lower() == '.pdf':
            output = arg
        else:
            folders.append(arg)

    if not folders:
        print("No input folders specified!")
        sys.exit(1)

    webp_multi_to_pdf(folders, output)

if __name__ == "__main__":
    main()