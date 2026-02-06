import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from PIL import Image
import torch

# 环境变量设置
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, help="e.g., 50, 75, 100")
    parser.add_argument("--dataset_type", type=str, choices=['utility', 'harm_3d'], required=True)
    parser.add_argument("--model_path", type=str, default="/path/to/dinov2/TRELLIS.2-4B")
    return parser.parse_args()

def prepare_config(model_path, variant):
    """动态准备并修改配置文件"""
    ckpt_dir = os.path.join(model_path, 'ckpts')
    base_json = os.path.join(ckpt_dir, 'ftss.json')
    target_json = os.path.join(ckpt_dir, f'ft-{variant}.json')
    
    # 1. 如果不存在则拷贝
    if not os.path.exists(target_json):
        print(f"Creating config: {target_json}")
        shutil.copy(base_json, target_json)
    
    # 2. 修改配置中的模型路径
    with open('config_ft.json', 'r') as f:
        config = json.load(f)
    
    # 修改具体的 flow_model 路径
    config['args']['models']['sparse_structure_flow_model'] = f'ckpts/ft-{variant}'
    
    with open('config_ft.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    return 'config_ft.json'

def get_image_files(directory, dataset_type):
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                if dataset_type == 'harm_3d':
                    # 严格过滤逻辑
                    if 'processed' in file and ('prototype_processed' in root or 'reimage_sd3.5' in root):
                        image_files.append(os.path.join(root, file))
                else:
                    # utility 逻辑：全部包含
                    image_files.append(os.path.join(root, file))
    return image_files

def main():
    args = parse_args()
    
    # 路径配置
    src_dir = f'/path/to/harm_3d_{args.dataset_type}' if args.dataset_type == 'utility' else '/path/to/harm_3d'
    dst_dir = f'/path/to//harm_3d_mesh_trellis2_{args.dataset_type}_{args.variant}'
    
    # 准备配置
    config_file_path = prepare_config(args.model_path, args.variant)

    # 初始化 Pipeline
    print(f"Loading TRELLIS with config: {config_file_path}")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        args.model_path, 
        config_file=config_file_path
    )
    pipeline.cuda()

    all_image_paths = get_image_files(src_dir, args.dataset_type)
    print(f"Found {len(all_image_paths)} images.")

    for i, img_path in enumerate(all_image_paths):
        try:
            rel_path = os.path.relpath(os.path.dirname(img_path), src_dir)
            target_folder = os.path.join(dst_dir, rel_path)
            target_file = os.path.join(target_folder, os.path.splitext(os.path.basename(img_path))[0] + ".obj")

            os.makedirs(target_folder, exist_ok=True)
            image = Image.open(img_path)

            mesh = pipeline.run(image, seed=1, pipeline_type='512')[0]
            mesh.simplify() 
            
            mesh_scene = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
                coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=1000000,
                texture_size=4096, remesh=True, remesh_band=1, remesh_project=0, verbose=False
            )
            mesh_scene.export(target_file)
            print(f"[{i+1}/{len(all_image_paths)}] Saved: {target_file}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    main()