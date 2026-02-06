import os
import sys
from pathlib import Path
from PIL import Image

# --- Environment Configuration ---
# Set these before importing torch/trellis related modules
os.environ['SPCONV_ALGO'] = 'native'  # 'native' is faster for one-off runs

# Add local trellis path
sys.path.append('./trellis')

try:
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils
except ImportError as e:
    print(f"Error importing TRELLIS modules: {e}")
    print("Please ensure you are running this script from the TRELLIS root directory.")
    sys.exit(1)

# --- Configuration ---
SRC_DIR = '/path/to/harm_3d'
DST_DIR = '/path/to//harm_3d_mesh_trellis'
MODEL_PATH = '/path/to/TRELLIS'

# Generation Parameters
SEED = 1
SIMPLIFY_RATIO = 0
TEXTURE_SIZE = 1024

def get_image_files(directory):
    """Recursively find all image files in directory containing 'processed'."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions and 'processed' in file:
                image_files.append(os.path.join(root, file))
    return image_files

def main():
    # 1. Initialize Pipeline
    print(f"Loading TRELLIS pipeline from {MODEL_PATH}...")
    try:
        pipeline = TrellisImageTo3DPipeline.from_pretrained(MODEL_PATH)
        pipeline.cuda()
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        return

    # 2. Collect all images
    print(f"Scanning {SRC_DIR}...")
    all_image_paths = get_image_files(SRC_DIR)
    total_images = len(all_image_paths)
    print(f"Found {total_images} images to process.")

    if total_images == 0:
        return

    # 3. Process Loop
    for i, img_path in enumerate(all_image_paths):
        try:
            # Determine Output Path
            # Maintain the same subdirectory structure
            rel_path = os.path.relpath(os.path.dirname(img_path), SRC_DIR)
            filename = os.path.splitext(os.path.basename(img_path))[0] + ".obj"
            
            target_folder = os.path.join(DST_DIR, rel_path)
            target_file = os.path.join(target_folder, filename)

            # Skip if file already exists
            if os.path.exists(target_file):
                print(f"[{i+1}/{total_images}] Skipping (Exists): {target_file}")
                continue

            print(f"[{i+1}/{total_images}] Processing: {img_path}")
            
            # Ensure target directory exists
            os.makedirs(target_folder, exist_ok=True)

            # Load Image
            image = Image.open(img_path)

            # Run Pipeline
            outputs = pipeline.run(
                image,
                seed=SEED
            )

            # Post-processing (Extract GLB/Mesh)
            # Using 'to_glb_pure' as requested, which returns a Trimesh Scene object
            mesh_scene = postprocessing_utils.to_glb_pure(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=SIMPLIFY_RATIO,
                texture_size=TEXTURE_SIZE,
            )

            # Export
            # The 'export' method infers format from extension (.obj)
            mesh_scene.export(target_file)
            print(f"Saved: {target_file}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()

    print("Processing complete.")

if __name__ == "__main__":
    main()