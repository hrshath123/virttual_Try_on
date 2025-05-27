import os
import argparse
import cv2
import numpy as np
from pathlib import Path
import subprocess

def generate_cloth_mask(input_path, output_path):
    """
    Generates a binary cloth mask for a given clothing image.
    """
    if not os.path.exists(input_path):
        print(f"Error: File not found → {input_path}")
        return

    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Unable to read the image → {input_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing
    kernel = np.ones((15, 15), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask
    mask = np.zeros_like(gray)

    # Keep contours above a threshold size
    min_area = 5000
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Save the final mask
    cv2.imwrite(output_path, mask)
    print(f"✅ Cloth mask saved at: {output_path}")

    return output_path

def clear_results_folder(results_folder):
    """Deletes all existing files in the results folder before processing."""
    if os.path.exists(results_folder):
        for filename in os.listdir(results_folder):
            file_path = os.path.join(results_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"❌ Error deleting {file_path}: {e}")
        print("🗑️ Cleared previous results from results/ folder.")

def update_test_pairs(image_folder, test_pairs_file, cloth_name):
    """
    Updates test_pairs.txt with the selected cloth and all models.
    """
    if not os.path.exists(image_folder):
        print(f"❌ ERROR: Image folder does not exist → {image_folder}")
        return

    model_images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))]

    if not model_images:
        print("⚠️ WARNING: No model images found in the image folder!")

    with open(test_pairs_file, "w") as f:
        for model in model_images:
            f.write(f"{model} {cloth_name}\n")
    print(f"✅ Updated test_pairs.txt: paired '{cloth_name}' with {len(model_images)} models.")

def main(cloth_path):
    # Resolve project base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define relative paths under base_dir
    image_folder     = os.path.join(base_dir, "datasets", "test", "image")
    cloth_mask_folder = os.path.join(base_dir, "datasets", "test", "cloth-mask")
    test_pairs_file  = os.path.join(base_dir, "datasets", "test", "test_pairs.txt")
    results_folder   = os.path.join(base_dir, "results")

    # Ensure necessary folders exist
    os.makedirs(cloth_mask_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    clear_results_folder(results_folder)

    if not os.path.exists(image_folder):
        print(f"❌ ERROR: Model-image folder does not exist → {image_folder}")
        return

    # Generate cloth mask
    cloth_name = Path(cloth_path).name
    cloth_mask_path = os.path.join(cloth_mask_folder, cloth_name)
    generate_cloth_mask(cloth_path, cloth_mask_path)

    # Debug: print image-folder contents
    print(f"📂 Checking image-folder: {image_folder}")
    print(f"✅ Exists? {os.path.exists(image_folder)}")
    print(f"📜 Contents: {os.listdir(image_folder)}")

    # Update test_pairs.txt
    update_test_pairs(image_folder, test_pairs_file, cloth_name)

    # Run test.py using the same Python interpreter
    print("🚀 Running test.py to apply virtual try-on…")
    python_executable = sys.executable  # ensures we use the same interpreter
    test_script_path = os.path.join(base_dir, "test.py")
    subprocess.run([python_executable, test_script_path, "--name", "virtual_tryon"],
                   cwd=base_dir)

    print("✅ Virtual try-on process complete! Results are in:", results_folder)

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("cloth_path", type=str, help="Path to the cloth image")
    args = parser.parse_args()
    main(args.cloth_path)
