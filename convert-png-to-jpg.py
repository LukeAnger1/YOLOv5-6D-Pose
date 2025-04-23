# Command to run python3 convert-png-to-jpg.py --input_dir "/home/rick/Desktop/nasa/yolo container/data/output/images" --output_dir "/home/rick/Desktop/nasa/yolo container/data/output/JPEGImages"

#!/usr/bin/env python3
"""
Script to convert all PNG files in a folder to PNG format.
Since the source and target formats are the same, this will essentially create copies.
"""

import os
import argparse
from PIL import Image
import concurrent.futures
from tqdm import tqdm

def convert_image(file_path, output_dir, output_suffix=""):
    """
    Convert a single image from PNG to JPG.
    """
    try:
        # Get the filename without extension and the extension
        filename, ext = os.path.splitext(os.path.basename(file_path))
        
        # Create the output filename with .jpg extension
        output_filename = f"{filename}{output_suffix}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        # Open the image and convert to RGB (JPG doesn't support transparency)
        img = Image.open(file_path).convert("RGB")
        img.save(output_path, 'JPEG')
        
        return True
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert PNG files to PNG format')
    parser.add_argument('--input_dir', help='Input directory containing PNG files')
    parser.add_argument('--output_dir', help='Output directory for converted files (default: input_dir/converted)')
    parser.add_argument('--suffix', default="", help='Suffix to add to converted filenames')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes for parallel conversion')
    
    args = parser.parse_args()
    
    # Set up input and output directories
    input_dir = os.path.abspath(args.input_dir)
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(input_dir, "converted")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all PNG files
    png_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))
    
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return
    
    print(f"Found {len(png_files)} PNG files to convert")
    
    # Process files in parallel with progress bar
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Create a list of futures
        futures = [executor.submit(convert_image, file_path, output_dir, args.suffix) 
                  for file_path in png_files]
        
        # Process as they complete with a progress bar
        successful = 0
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Converting images"):
            if _.result():
                successful += 1
    
    print(f"Conversion complete: {successful} of {len(png_files)} files converted successfully")
    print(f"Converted files saved to: {output_dir}")

if __name__ == "__main__":
    main()