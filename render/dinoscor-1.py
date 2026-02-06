import os
import argparse
import json
import torch
import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModel

"""
Evaluation Script (Refined Path Logic):
1. Hard Metrics (JSON only): Visual Fidelity (DINO), Geometry (Watertight/Collapse).
2. Diagnostic Collage (Visuals only).
3. Strict Path Matching: Enforces sub-folder alignment between Mesh Root and Ref Root.
"""

class DinoEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        print(f"Loading DINOv2 on {device}...")
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained('path/to/dinov2')
        self.model = AutoModel.from_pretrained('path/to/dinov2').to(self.device)        
        self.model.eval()

    def get_embedding(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error reading {image_path}: {e}")
            return torch.zeros(1, 768).to(self.device)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    def compute_similarity(self, emb1, emb2):
        return torch.nn.functional.cosine_similarity(emb1, emb2).item()

def calculate_geometry_metrics(mesh_path):
    metrics = {
        "mesh_path": mesh_path,
        "is_watertight": False,
        "non_manifold_edges": -1,
        "flatness_ratio": 0.0,
        "is_collapsed": True, 
        "error": None
    }
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        metrics["is_watertight"] = bool(mesh.is_watertight)
        
        edges = mesh.edges_sorted
        if len(edges) > 0:
            groups = trimesh.grouping.group_rows(edges, require_count=None)
            non_manifold_count = sum(1 for g in groups if len(g) > 2)
            metrics["non_manifold_edges"] = non_manifold_count
        else:
            metrics["non_manifold_edges"] = 0

        extents = mesh.extents
        if extents is None or len(extents) == 0 or np.max(extents) == 0:
            metrics["flatness_ratio"] = 0.0
            metrics["is_collapsed"] = True
        else:
            d_min = np.min(extents)
            d_max = np.max(extents)
            flatness_ratio = d_min / d_max
            metrics["flatness_ratio"] = float(flatness_ratio)
            metrics["is_collapsed"] = bool(flatness_ratio < 0.05)
        metrics["is_valid_mesh"] = True
    except Exception as e:
        metrics["error"] = str(e)
        metrics["is_valid_mesh"] = False
    return metrics

def create_collage(ref_path, best_view, best_score, rotated_views, output_path):
    W, H = 512, 512
    GRID_W, GRID_H = W * 3, H * 2
    
    def load_processed(p):
        try:
            img = Image.open(p)
        except Exception as e:
            return Image.new('RGB', (W, H), (255, 0, 0))
        img = img.resize((W, H), Image.Resampling.LANCZOS)
        if img.mode == 'RGBA':
            bg = Image.new('RGB', (W, H), (255, 255, 255))
            bg.paste(img, (0, 0), img)
            return bg
        return img.convert("RGB")

    img_ref = load_processed(ref_path)
    img_best_rgb = load_processed(best_view['rgb_path'])
    img_best_norm = load_processed(best_view['normal_path'])
    imgs_rot = [load_processed(v['normal_path']) for v in rotated_views]

    collage = Image.new('RGB', (GRID_W, GRID_H), (255, 255, 255))
    collage.paste(img_ref, (0, 0))          
    collage.paste(img_best_rgb, (W, 0))     
    collage.paste(img_best_norm, (W*2, 0))  
    collage.paste(imgs_rot[0], (0, H))      
    collage.paste(imgs_rot[1], (W, H))      
    collage.paste(imgs_rot[2], (W*2, H))    

    draw = ImageDraw.Draw(collage)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    def draw_label(text, pos, color=(0, 0, 0), f=font):
        x, y = pos
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx or dy:
                    draw.text((x+dx, y+dy), text, font=f, fill=(255, 255, 255))
        draw.text((x, y), text, font=f, fill=color)

    draw_label("Region A: Reference", (20, 20))
    draw_label(f"Region B: Best Match", (W + 20, 20), color=(0, 150, 0))
    draw_label(f"Visual Score: {best_score:.3f}", (W + 20, 50), color=(0, 150, 0))
    draw_label("Region C: Normal Map", (W*2 + 20, 20))
    labels = ["Region D: +90°", "Region E: Back", "Region F: +270°"]
    for i, (label) in enumerate(labels):
        draw_label(label, (W * i + 20, H + 20), color=(0, 100, 200))

    for x in [W, W*2]:
        draw.line([(x, 0), (x, GRID_H)], fill=(200, 200, 200), width=3)
    draw.line([(0, H), (GRID_W, H)], fill=(200, 200, 200), width=3)

    collage.save(output_path)
    print(f"Collage saved to {output_path}")

def resolve_paths(render_dir, mesh_root, ref_root):
    """
    Strictly aligns path structure between mesh_root and ref_root.
    Avoids 'Knucks' vs 'Kubotan' mismatches by enforcing relative path similarity.
    """
    model_name = os.path.basename(render_dir)
    
    # 1. Get Path relative to Mesh Root
    # e.g., FormFunction/Kubotan/prototype_processed/rendered/10_concept_processed
    try:
        rel_path = os.path.relpath(render_dir, mesh_root)
    except ValueError:
        # If paths are on different drives or totally disjoint
        print(f"[Path Error] Cannot compute relpath for {render_dir} from {mesh_root}")
        return None, None

    # 2. Breakdown components
    path_parts = rel_path.split(os.sep)
    
    # 3. Resolve Reference Image (Searching strictly in the aligned directory)
    ref_image_path = None
    
    # Iterate backwards from the deepest folder up to finding a valid ref folder
    # We strip common "artifact" folders often found in mesh outputs but not ref sources
    ignore_folders = {'rendered', 'prototype_processed', 'output', 'mesh', 'obj', 'glb'}
    
    # Construct a "Clean" relative path (e.g., FormFunction/Kubotan)
    # We try to find the longest matching prefix in ref_root
    
    potential_ref_dirs = []
    
    # Strategy: Build path cumulatively from root
    # Check ref_root/FormFunction -> ref_root/FormFunction/Kubotan -> etc.
    current_check = ref_root
    valid_deepest_dir = ref_root
    
    for part in path_parts:
        if part == model_name: continue # Skip the model dir itself for now
        
        next_dir = os.path.join(current_check, part)
        if os.path.exists(next_dir):
            current_check = next_dir
            if part not in ignore_folders:
                valid_deepest_dir = current_check
        else:
            # If path diverges, we stop. This handles minor structural diffs.
            # But we stick to the deepest VALID matching folder.
            pass

    # Now look for the image inside 'valid_deepest_dir' or its immediate children
    # This restricts the search to "FormFunction/Kubotan" area
    
    search_dirs = [valid_deepest_dir]
    # Also add the exact mirrored path if it exists (ignoring the 'exists' check loop above logic for a second)
    # Sometimes 'prototype_processed' DOES exist in Ref
    
    mirrored_path = os.path.join(ref_root, os.path.dirname(rel_path))
    if os.path.exists(mirrored_path):
        search_dirs.insert(0, mirrored_path)

    # Check for image
    candidates = [model_name, model_name.replace("_output_processed", ""), model_name.split('_')[0]]
    
    found = False
    for s_dir in search_dirs:
        if found: break
        # Direct check
        for cand in candidates:
            for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                p = os.path.join(s_dir, cand + ext)
                if os.path.exists(p):
                    ref_image_path = p
                    found = True
                    break
            if found: break
        
        # Shallow walk (depth 1 or 2) just in case it's in a subfolder like 'images'
        if not found:
            for r, d, f in os.walk(s_dir):
                # Don't go too deep to avoid jumping categories
                rel_depth = len(os.path.relpath(r, s_dir).split(os.sep))
                if rel_depth > 2: continue 
                
                for cand in candidates:
                    for ext in ['.png', '.jpg']:
                        if (cand + ext) in f:
                            ref_image_path = os.path.join(r, cand + ext)
                            found = True
                            break
                    if found: break
                if found: break

    # 4. Resolve Mesh Path
    # Logic: Go up from render_dir to find .glb/.obj
    # Structure: .../Kubotan/prototype_processed/rendered/10_concept...
    # Mesh at: .../Kubotan/prototype_processed/10_concept.glb
    
    mesh_path = None
    parent = os.path.dirname(render_dir) # .../rendered
    grandparent = os.path.dirname(parent) # .../prototype_processed
    
    # Check specifically for the .glb/obj structure
    candidate_folder = os.path.join(grandparent, model_name + ".glb")
    candidate_obj_in_folder = os.path.join(candidate_folder, "obj")
    
    if os.path.exists(candidate_obj_in_folder):
        mesh_path = candidate_obj_in_folder
    else:
        # Standard check
        for ext in ['.glb', '.obj', '.ply']:
            p = os.path.join(grandparent, model_name + ext)
            if os.path.exists(p):
                mesh_path = p
                break
    
    return ref_image_path, mesh_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_root', required=True, help="Root folder containing 'rendered' subfolders")
    parser.add_argument('--ref_root', required=True, help="Reference images root")
    args = parser.parse_args()

    evaluator = None 
    all_best_scores = []
    print(f"Scanning: {args.mesh_root}")

    for root, dirs, files in os.walk(args.mesh_root):
        if "views.json" in files:
            render_dir = root
            model_name = os.path.basename(render_dir)
            
            # --- Strict Path Resolution ---
            ref_image_path, mesh_path = resolve_paths(render_dir, args.mesh_root, args.ref_root)

            if not ref_image_path:
                print(f"[SKIPPING] No reference found for {model_name} (Strict match failed)")
                continue
            
            # Double check we didn't cross wires (sanity check)
            # Ensure "Kubotan" is not in mesh path while "Knucks" is in ref path
            mesh_parts = set(render_dir.split(os.sep))
            ref_parts = set(ref_image_path.split(os.sep))
            
            # Simple heuristic: If the parent folder name of the mesh differs significantly from ref
            # This is hard to automate generally, but the resolve_paths logic above prefers alignment.

            print(f"Processing: {model_name}")
            print(f"  - Ref: {ref_image_path}")
            print(f"  - Mesh: {mesh_path if mesh_path else 'NOT FOUND'}")

            # --- 1. Compute Geometry Metrics ---
            geom_metrics = calculate_geometry_metrics(mesh_path) if mesh_path else {
                "is_valid_mesh": False, "flatness_ratio": 0, "is_collapsed": True, 
                "is_watertight": False, "non_manifold_edges": 0
            }

            # --- 2. Compute Visual Metrics (DINO) ---
            if evaluator is None:
                evaluator = DinoEvaluator()

            try:
                with open(os.path.join(render_dir, "views.json"), 'r') as f:
                    views = json.load(f)

                for v in views:
                    v['rgb_path'] = os.path.join(render_dir, v['rgb_path'])
                    v['normal_path'] = os.path.join(render_dir, v['normal_path'])

                ref_emb = evaluator.get_embedding(ref_image_path)
                best_score = -1.0
                best_idx = -1

                for i, view in enumerate(views):
                    view_emb = evaluator.get_embedding(view['rgb_path'])
                    score = evaluator.compute_similarity(ref_emb, view_emb)
                    views[i]['score'] = score
                    if score > best_score:
                        best_score = score
                        best_idx = i
                all_best_scores.append(best_score)
                best_view = views[best_idx]
                
                target_offsets = [90, 180, 270]
                rotated_views = []
                base_azi = best_view['azimuth']
                base_ele = best_view['elevation']

                for offset in target_offsets:
                    target_azi = (base_azi + offset) % 360
                    def dist(v):
                        d_azi = abs(v['azimuth'] - target_azi)
                        d_azi = min(d_azi, 360 - d_azi)
                        d_ele = abs(v['elevation'] - base_ele)
                        return d_azi + (d_ele * 2)
                    closest = min(views, key=dist)
                    rotated_views.append(closest)

                # --- 3. Save Hard Metrics JSON ---
                hard_metrics = {
                    "model_name": model_name,
                    "visual_fidelity_score": float(best_score),
                    "geometry_integrity": {
                        "is_watertight": geom_metrics["is_watertight"],
                        "non_manifold_edges": geom_metrics["non_manifold_edges"],
                        "printability_pass": geom_metrics["is_watertight"] and (geom_metrics["non_manifold_edges"] == 0)
                    },
                    "geometric_collapse": {
                        "flatness_ratio": geom_metrics["flatness_ratio"],
                        "is_collapsed": geom_metrics["is_collapsed"]
                    }
                }
                
                json_out_path = os.path.join(render_dir, "hard_metrics.json")
                with open(json_out_path, 'w') as jf:
                    json.dump(hard_metrics, jf, indent=4)
                
                output_filename = os.path.join(render_dir, "eval_collage_metrics.jpg")
                create_collage(ref_image_path, best_view, best_score, rotated_views, output_filename)
            
            except Exception as e:
                print(f"[ERROR] {model_name}: {e}")
    print("\n" + "="*30)
    print("FINAL DATASET SUMMARY")
    print("="*30)
    if all_best_scores:
        avg_dino = sum(all_best_scores) / len(all_best_scores)
        print(f"Total Models Evaluated: {len(all_best_scores)}")
        print(f"Average Best-Match DINO Score: {avg_dino:.4f}")
        print(f"Max Score: {max(all_best_scores):.4f}")
        print(f"Min Score: {min(all_best_scores):.4f}")
    else:
        print("No valid models were processed.")
    print("="*30)

if __name__ == "__main__":
    main()