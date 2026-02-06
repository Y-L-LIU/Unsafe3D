import os
import sys
import math
import argparse
from pathlib import Path
from PIL import Image

# 保持原有的路径插入
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Hunyuan3D Batch Processing")
    parser.add_argument("--variant", type=str, required=True, help="Model variant (e.g., unlearnall-50)")
    parser.add_argument("--dataset_type", type=str, choices=['utility', 'harm_3d'], required=True)
    parser.add_argument("--model_path", type=str, default='/path/to/hunyuan3d-2.1')
    parser.add_argument("--batch_size", type=int, default=12)
    return parser.parse_args()

def get_image_files(directory, dataset_type):
    """根据数据集类型执行不同的过滤逻辑"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                # Logic for harm_3d
                if dataset_type == 'harm_3d':
                    if 'processed' in file and ('prototype_processed' in root or 'reimage_sd3.5' in root):
                        image_files.append(os.path.join(root, file))
                # Logic for utility (no extra filtering)
                else:
                    image_files.append(os.path.join(root, file))
    return image_files

def main():
    args = parse_args()
    
    # 设置路径映射
    dst_dir = f'/path/to/dataset/harm_3d_mesh_hunyuan2_{args.dataset_type}_{args.variant}'
    # 获取源路径
    src_dir = '/path/to/dataset/harm_3d_utility' if args.dataset_type == 'utility' else '/path/to/dataset/harm_3d'
    print(f"--- Config ---")
    print(f"Variant: {args.variant} | Dataset: {args.dataset_type}")
    print(f"Source: {src_dir} | Destination: {dst_dir}")

    # 初始化 Pipeline
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path, 
        variant=args.variant
    )

    all_image_paths = get_image_files(src_dir, args.dataset_type)
    total_images = len(all_image_paths)
    print(f"Found {total_images} images.")

    if total_images == 0: return

    num_batches = math.ceil(total_images / args.batch_size)
    for i in range(num_batches):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, total_images)
        batch_paths = all_image_paths[start_idx:end_idx]
        batch_images = []
        valid_paths = []

        for path in batch_paths:
            try:
                batch_images.append(Image.open(path).convert("RGBA"))
                valid_paths.append(path)
            except Exception as e: print(f"Error: {e}")

        if not batch_images: continue

        try:
            meshes = pipeline_shapegen(
                image=batch_images, steps=50, guidance_scale=7.5,
                seed=1234, octree_resolution=384, mc_level=0.0,
                mc_algo='mc', check_box_rembg=False, num_chunks=100000
            )

            for img_path, mesh in zip(valid_paths, meshes):
                rel_path = os.path.relpath(os.path.dirname(img_path), src_dir)
                filename = os.path.splitext(os.path.basename(img_path))[0] + ".obj"
                target_folder = os.path.join(dst_dir, rel_path)
                os.makedirs(target_folder, exist_ok=True)
                mesh.export(os.path.join(target_folder, filename))
                
        except Exception as e:
            print(f"Batch failed: {e}")

if __name__ == "__main__":
    main()