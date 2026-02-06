import argparse
import os
import sys
import math
from glob import glob
from typing import Any, Union, List
from pathlib import Path

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image

# Add specific paths as requested
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('/home/lyl/projects/harm3d/TripoSG')

try:
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    from image_process import prepare_image
    from briarmbg import BriaRMBG
except ImportError as e:
    print(f"Error importing TripoSG modules: {e}")
    print("Please ensure you are running this script from the correct environment and location.")
    sys.exit(1)

import pymeshlab

# --- Configuration ---
SRC_DIR = '/path/to//harm_3d'
# Using a distinct output folder for TripoSG results
DST_DIR = '/path/to/harm_3d_mesh_triposg' 
MODEL_PATH = '/path/to/TripoSG'
RMBG_PATH = 'pretrained_weights/RMBG-1.4'

# Fixed Parameters
SEED = 42
STEPS = 50
GUIDANCE = 7.0
FACES = -1

@torch.no_grad()
def run_triposg(
    pipe: Any,
    image_input: Union[str, Image.Image],
    rmbg_net: Any,
    seed: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    faces: int = -1,
) -> trimesh.Scene:

    img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)

    outputs = pipe(
        image=img_pil,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).samples[0]
    
    # Extract the mesh from the output
    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))

    if faces > 0:
        mesh = simplify_mesh(mesh, faces)

    return mesh

def mesh_to_pymesh(vertices, faces):
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    return ms

def pymesh_to_trimesh(mesh):
    verts = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    return trimesh.Trimesh(vertices=verts, faces=faces)

def simplify_mesh(mesh: trimesh.Trimesh, n_faces):
    if mesh.faces.shape[0] > n_faces:
        ms = mesh_to_pymesh(mesh.vertices, mesh.faces)
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
        return pymesh_to_trimesh(ms.current_mesh())
    else:
        return mesh

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    # 1. Initialization
    print("Initializing models...")
    # Ensure weights are present
    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=MODEL_PATH)
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=RMBG_PATH)

    # Init RMBG model
    rmbg_net = BriaRMBG.from_pretrained(RMBG_PATH).to(device)
    rmbg_net.eval() 

    # Init TripoSG pipeline
    pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained(MODEL_PATH).to(device, dtype)
    
    # 2. Collect all images
    print(f"Scanning {SRC_DIR}...")
    all_image_paths = get_image_files(SRC_DIR)
    total_images = len(all_image_paths)
    print(f"Found {total_images} images to process.")

    if total_images == 0:
        return

    # 3. Process loop
    # Note: While the pipeline *can* support batching in theory, the provided wrapper 'run_triposg'
    # handles specific image preprocessing and returns a single mesh. To ensure stability and 
    # use the exact provided logic, we iterate through the list.
    
    for i, img_path in enumerate(all_image_paths):
        try:
            # Determine Output Path
            rel_path = os.path.relpath(os.path.dirname(img_path), SRC_DIR)
            filename = os.path.splitext(os.path.basename(img_path))[0] + ".obj"
            
            target_folder = os.path.join(DST_DIR, rel_path)
            target_file = os.path.join(target_folder, filename)

            # Skip if exists
            if os.path.exists(target_file):
                print(f"[{i+1}/{total_images}] Skipping (Exists): {target_file}")
                continue

            print(f"[{i+1}/{total_images}] Processing: {img_path}")
            
            # Ensure target directory exists
            os.makedirs(target_folder, exist_ok=True)

            # Load Image
            # img = Image.open(img_path).convert("RGB")

            # Run Inference
            mesh = run_triposg(
                pipe=pipe,
                image_input=img_path,
                rmbg_net=rmbg_net,
                seed=SEED,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                faces=FACES,
            )

            # Export
            # Using standard .obj export. TripoSG output often benefits from .glb for vertex colors,
            # but .obj was requested in the directory structure example.
            mesh.export(target_file)
            print(f"Saved: {target_file}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()

    print("Processing complete.")

if __name__ == "__main__":
    main()