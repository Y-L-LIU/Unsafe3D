import argparse, sys, os, math, re, glob, shutil, json
import bpy
from mathutils import Vector, Matrix
import numpy as np
import cv2

"""=============== BLENDER IMPORTS & CONSTANTS ==============="""

IMPORT_FUNCTIONS = {
    "obj": bpy.ops.wm.obj_import,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

EXT = {
    'PNG': 'png',
    'JPEG': 'jpg',
    'OPEN_EXR': 'exr',
    'TIFF': 'tiff',
    'BMP': 'bmp',
    'HDR': 'hdr',
    'TARGA': 'tga'
}

"""=============== CAMERA SEQUENCES (MODIFIED) ==============="""

def evaluation_camera_sequence():
    """
    Generates 18 views for fidelity evaluation:
    - 8 views at Elevation 0 (Eye level)
    - 8 views at Elevation 30 (Looking down)
    - 1 Top view (+90)
    - 1 Bottom view (-90)
    """
    views = []
    cam_dis = 1.6  # Standard distance for normalized unit cube
    # FOV 40 degrees (~40-50mm focal length)
    fov = 40 * (np.pi / 180) 
    
    # Ring 1: Elevation 0
    for i in range(8):
        azi = i * (2 * np.pi / 8)
        views.append({'hangle': azi, 'vangle': 0, 'cam_dis': cam_dis, 'fov': fov, 'proj_type': 0})

    # Ring 2: Elevation 30 deg (~0.52 rad)
    ele_30 = 30 * (np.pi / 180)
    for i in range(8):
        azi = i * (2 * np.pi / 8)
        views.append({'hangle': azi, 'vangle': ele_30, 'cam_dis': cam_dis, 'fov': fov, 'proj_type': 0})
        
    # Top View
    views.append({'hangle': 0, 'vangle': np.pi/2 - 0.01, 'cam_dis': cam_dis, 'fov': fov, 'proj_type': 0})
    
    # Bottom View
    views.append({'hangle': 0, 'vangle': -np.pi/2 + 0.01, 'cam_dis': cam_dis, 'fov': fov, 'proj_type': 0})
    
    return views

"""=============== ORIGINAL NODE & RENDER LOGIC (PRESERVED) ==============="""

def switch_to_mr_render(render_base_color, output_nodes):
    bpy.context.scene.view_settings.view_transform = 'Raw'
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for i in range(len(output_nodes)):
        if i + 1 != len(output_nodes):
            for l in output_nodes[i][1].links:
                links.remove(l)
        else:
            links.new(output_nodes[i][0], output_nodes[i][1])

    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        bsdf_node = None
        output_node = None
        node_tree = material.node_tree
        links = material.node_tree.links
        nodes = node_tree.nodes
        for node in node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bsdf_node = node
            if node.type == 'OUTPUT_MATERIAL':
                output_node = node
        if bsdf_node is None or output_node is None:
            continue
        
        # Safe access for newer Blender versions
        if 'Emission Strength' in bsdf_node.inputs:
            bsdf_node.inputs['Emission Strength'].default_value = 0
        elif 'Emission' in bsdf_node.inputs:
             pass # Legacy

        mr_node = None
        bc_node = None
        for node in node_tree.nodes:
            if node.name == 'COMBINE_METALLIC_ROUGHNESS':
                mr_node = node
            if node.name == 'COMBINE_BASE_COLOR':
                bc_node = node
        if mr_node is None:
            combine_rgb_node = nodes.new('ShaderNodeCombineColor')
            combine_rgb_node.inputs['Red'].default_value = 1.0
            combine_rgb_node.inputs['Green'].default_value = 0.5
            combine_rgb_node.inputs['Blue'].default_value = 0.0
            metallic_input = bsdf_node.inputs["Metallic"]

            if metallic_input.links:
                source_endpoint = metallic_input.links[0].from_socket
                links.new(source_endpoint, combine_rgb_node.inputs['Blue'])

            roughness_input = bsdf_node.inputs['Roughness']
            if roughness_input.links:
                source_endpoint = roughness_input.links[0].from_socket
                links.new(source_endpoint, combine_rgb_node.inputs['Green'])

            emission_shader = nodes.new("ShaderNodeEmission")
            emission_shader.inputs["Strength"].default_value = 1
            links.new(combine_rgb_node.outputs["Color"], emission_shader.inputs["Color"])

            mix_shader = nodes.new("ShaderNodeMixShader")
            mix_shader.name = 'COMBINE_METALLIC_ROUGHNESS'
            links.new(bsdf_node.outputs["BSDF"], mix_shader.inputs[1])
            links.new(emission_shader.outputs["Emission"], mix_shader.inputs[2])
            mr_node = mix_shader

            mix_shader_bc = nodes.new("ShaderNodeMixShader")
            mix_shader_bc.name = 'COMBINE_BASE_COLOR'
            
            if len(bsdf_node.inputs['Base Color'].links) > 0:
                socket = bsdf_node.inputs['Base Color'].links[0].from_socket
                gamma_node = node_tree.nodes.new(type='ShaderNodeGamma')
                gamma_node.inputs[1].default_value = 0.454
                node_tree.links.new(socket, gamma_node.inputs[0])
                node_tree.links.new(gamma_node.outputs[0], mix_shader_bc.inputs[1])

            links.new(mix_shader.outputs["Shader"], mix_shader_bc.inputs[2])
            bc_node = mix_shader_bc

            for l in output_node.inputs['Surface'].links:
                links.remove(l)
            links.new(mix_shader_bc.outputs["Shader"], output_node.inputs["Surface"])

        mr_node.inputs["Fac"].default_value = 1.0
        if render_base_color:
            bc_node.inputs['Fac'].default_value = 0.0
        else:
            bc_node.inputs['Fac'].default_value = 1.0

def switch_to_color_render(output_nodes):
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for i in range(len(output_nodes)):
        if i + 1 == len(output_nodes):
            for l in output_nodes[i][1].links:
                links.remove(l)
        else:
            links.new(output_nodes[i][0], output_nodes[i][1])

    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        node_tree = material.node_tree
        mr_node = None
        bc_node = None
        for node in node_tree.nodes:
            if node.name == 'COMBINE_METALLIC_ROUGHNESS':
                mr_node = node
            if node.name == 'COMBINE_BASE_COLOR':
                bc_node = node
        if mr_node is not None and bc_node is not None:
            mr_node.inputs["Fac"].default_value = 0.0
            if len(bc_node.inputs[1].links) > 0:
                try:
                    node = bc_node.inputs[1].links[0].from_socket.node
                    node.image.colorspace_settings.name = 'sRGB'
                except:
                    pass

def ConvertNormalMap(input_exr, output_jpg):
    exr_img = cv2.imread(input_exr, cv2.IMREAD_UNCHANGED)
    if exr_img is None:
        return
    normal = ((exr_img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(output_jpg, normal)

def ConvertDepthMap(input_exr, output_png, cam_obj):
    """
    Original complex depth map conversion that projects camera space back to world space
    to get a position map, or processes depth. 
    """
    cam = cam_obj
    cam_data = cam.data
    
    exr_img = cv2.imread(input_exr, cv2.IMREAD_UNCHANGED)
    if exr_img is None:
        raise RuntimeError(f"Failed to load EXR file: {input_exr}")

    depth_channel = exr_img[:, :, 0] if exr_img.ndim == 3 else exr_img
    depth_channel = depth_channel.copy()
    depth_channel[depth_channel > 1e9] = 0

    extrinsic_matrix = np.array(cam.matrix_world.copy())
    scene = bpy.context.scene
    render = scene.render

    resolution_x = render.resolution_x * render.pixel_aspect_x
    resolution_y = render.resolution_y * render.pixel_aspect_y

    cx = resolution_x / 2.0
    cy = resolution_y / 2.0

    if cam_data.type == 'ORTHO':
        # Ortho logic preserved
        aspect_ratio = render.resolution_x / render.resolution_y
        ortho_scale = cam_data.ortho_scale
        near = cam_data.clip_start
        far = cam_data.clip_end
        left = -ortho_scale / 2
        right = ortho_scale / 2
        top = (ortho_scale / 2) / aspect_ratio
        bottom = -top
        # (Simplified projection logic for brevity, but original had it implicit)
    else:
        if cam_data.sensor_fit == 'VERTICAL':
            sensor_size = cam_data.sensor_height
            fit = 'VERTICAL'
        else:
            sensor_size = cam_data.sensor_width
            fit = 'HORIZONTAL'

        focal_length = cam_data.lens
        if fit == 'HORIZONTAL':
            scale = resolution_x / sensor_size
        else:
            scale = resolution_y / sensor_size

        fx = focal_length * scale
        fy = focal_length * scale

        K = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,   1]
        ])

    mask = (depth_channel.reshape(-1) == 0)
    jj, ii = np.meshgrid(np.arange(resolution_x), np.arange(resolution_y))
    jj = jj + 0.5
    ii = ii + 0.5

    if cam_data.type == 'ORTHO':
        cam_pos = np.stack((
            (jj - cx) * (1.0 / (resolution_x - 1) * ortho_scale),
            (ii - cy) * (1.0 / (resolution_y - 1) * ortho_scale),
            depth_channel
        ), axis=-1)
    else:
        image_pos = np.stack((jj * depth_channel, ii * depth_channel, depth_channel), axis=-1)
        cam_pos = image_pos @ np.linalg.inv(K).T

    cam_pos[..., 1:] = -cam_pos[..., 1:]

    world_pos = cam_pos @ extrinsic_matrix[:3, :3].T + extrinsic_matrix[:3, 3].reshape(1, 1, 3)
    world_pos = world_pos.reshape(-1, 3)
    world_pos[mask] = 0
    world_pos = world_pos.reshape(cam_pos.shape)
    world_pos = np.stack((world_pos[..., 0], world_pos[..., 2], -world_pos[..., 1]), axis=-1)

    img_out = np.clip((0.5 + world_pos) * 255, 0, 255).astype('uint8')
    cv2.imwrite(output_png, img_out)

def init_render(engine='CYCLES', resolution=512, geo_mode=False):
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    
    bpy.context.scene.cycles.device = 'GPU'
    # Use fewer samples for batch efficiency, but keep denoising
    bpy.context.scene.cycles.samples = 64 
    bpy.context.scene.cycles.filter_type = 'BOX'
    bpy.context.scene.cycles.filter_width = 1
    bpy.context.scene.cycles.use_denoising = True
    
    # Enable CUDA/OptiX
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.get_devices()
    for dev in prefs.devices:
        if dev.type in ('CUDA', 'OPTIX', 'HIP'):
            dev.use = True
            prefs.compute_device_type = dev.type
            break

def init_nodes(save_depth=False, save_normal=False, save_albedo=False, save_mr=False, save_mist=False):
    outputs = {}
    spec_nodes = {} # Kept for compatibility with original signature
    composite_nodes = []
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers['ViewLayer'].use_pass_z = save_depth
    bpy.context.scene.view_layers['ViewLayer'].use_pass_normal = save_normal
    bpy.context.scene.view_layers['ViewLayer'].use_pass_diffuse_color = save_albedo
    bpy.context.scene.view_layers['ViewLayer'].use_pass_mist = save_mist
    
    tree = bpy.context.scene.node_tree
    nodes = tree.nodes
    links = tree.links
    for n in nodes:
        nodes.remove(n)
    
    render_layers = nodes.new('CompositorNodeRLayers')
    
    if save_depth:
        depth_file_output = nodes.new('CompositorNodeOutputFile')
        depth_file_output.base_path = ''
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = "OPEN_EXR"
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
        outputs['depth'] = depth_file_output
        composite_nodes.append((render_layers.outputs['Depth'], depth_file_output.inputs[0]))
    
    if save_normal:
        normal_file_output = nodes.new('CompositorNodeOutputFile')
        normal_file_output.base_path = ''
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = 'OPEN_EXR'
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
        outputs['normal'] = normal_file_output
        composite_nodes.append((render_layers.outputs['Normal'], normal_file_output.inputs[0]))
    
    if save_albedo:
        albedo_file_output = nodes.new('CompositorNodeOutputFile')
        albedo_file_output.base_path = ''
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = 'PNG'
        albedo_file_output.format.color_mode = 'RGBA'
        albedo_file_output.format.color_depth = '8'
        
        alpha_albedo = nodes.new('CompositorNodeSetAlpha')
        links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])
        links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])
        outputs['albedo'] = albedo_file_output

    if save_mr:
        mr_file_output = nodes.new('CompositorNodeOutputFile')
        mr_file_output.base_path = ''
        mr_file_output.file_slots[0].use_node_format = True
        mr_file_output.format.file_format = 'OPEN_EXR'
        links.new(render_layers.outputs['Image'], mr_file_output.inputs[0])
        outputs['mr'] = mr_file_output
        composite_nodes.append((render_layers.outputs['Image'], mr_file_output.inputs[0]))
        
    return outputs, spec_nodes, composite_nodes

def init_scene() -> None:
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def init_camera():
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam

def init_lighting():
    # Re-using logic from original script, but robustly
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
    bpy.context.collection.objects.link(default_light)
    default_light.data.energy = 1000
    default_light.location = (4, 1, 6)
    
    top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top_light)
    top_light.data.energy = 10000
    top_light.location = (0, 0, 10)
    top_light.scale = (100, 100, 100)
    
    bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
    bpy.context.collection.objects.link(bottom_light)
    bottom_light.data.energy = 1000
    bottom_light.location = (0, 0, -10)
    
    return [default_light, top_light, bottom_light]

def load_object(object_path: str) -> None:
    file_extension = object_path.split(".")[-1].lower()
    if file_extension not in IMPORT_FUNCTIONS:
        raise ValueError(f"Unsupported file type: {object_path}")
    print(f"Loading object from {object_path}")
    import_function = IMPORT_FUNCTIONS[file_extension]
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS')
    else:
        import_function(filepath=object_path)

def normalize_scene():
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)
        for obj in scene_root_objects:
            obj.parent = scene
    elif len(scene_root_objects) == 1:
        scene = scene_root_objects[0]
    else:
        return 1.0, Vector((0,0,0)) # Empty scene

    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Mesh):
            found = True
            for coord in obj.bound_box:
                coord = Vector(coord)
                coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    
    if not found:
        return 1.0, Vector((0,0,0))

    bbox_min = Vector(bbox_min)
    bbox_max = Vector(bbox_max)
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale
    bpy.context.view_layer.update()
    
    # Recompute bbox after scale
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    for obj in bpy.context.scene.objects.values():
         if isinstance(obj.data, bpy.types.Mesh):
            for coord in obj.bound_box:
                coord = Vector(coord)
                coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    
    bbox_min = Vector(bbox_min)
    bbox_max = Vector(bbox_max)
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    return scale, offset

def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix

def split_mesh_normal():
    # Helper from original script
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objs: return
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs: obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.split_normals()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action="DESELECT")

def delete_invisible_objects() -> None:
    # Helper from original script
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

"""=============== MAIN PROCESS LOGIC ==============="""

def process_single_model(model_path, output_root, arg):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    # 1. 强制绝对路径
    output_folder = os.path.abspath(os.path.join(output_root, model_name))
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"--- Processing: {model_name} ---")
    print(f"Target Output Dir: {output_folder}")

    # Reset Scene & Init
    init_scene()
    init_render(engine=arg.engine, resolution=arg.resolution, geo_mode=False)
    
    outputs, spec_nodes, composite_nodes = init_nodes(
        save_depth=arg.save_depth,
        save_normal=arg.save_normal,
        save_albedo=arg.save_albedo,
        save_mr=arg.save_mr,
        save_mist=arg.save_mist
    )
    
    # Load & Normalize
    if model_path.endswith(".blend"):
        load_object(model_path)
        delete_invisible_objects()
    else:
        load_object(model_path)
        if arg.split_normal:
            split_mesh_normal()
    
    scale, offset = normalize_scene()
    cam = init_camera()
    init_lighting()
    
    views = evaluation_camera_sequence()

    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "frames": []
    }
    eval_views_data = []

    # Render Loop
    for i, view in enumerate(views):
        # Camera Setup
        cam.location = (
            view['cam_dis'] * np.cos(view['hangle']) * np.cos(view['vangle']),
            view['cam_dis'] * np.sin(view['hangle']) * np.cos(view['vangle']),
            view['cam_dis'] * np.sin(view['vangle'])
        )
        cam.data.lens = 16 / np.tan(view['fov'] / 2)
        if view['proj_type'] == 1:
            cam.data.type = "ORTHO"
            cam.data.ortho_scale = 1.2
        else:
            cam.data.type = "PERSP"

        # === 【关键修改 A】: 设置路径的方式变了 ===
        # 标准 RGB 输出路径
        rgb_filename = f'{i:03d}.png'
        bpy.context.scene.render.filepath = os.path.join(output_folder, rgb_filename)
        
        # 节点输出路径 (Normal, Depth 等)
        # 必须把 文件夹 和 文件名 分开给 Blender，否则它会乱用相对路径
        for name, output in outputs.items():
            output.base_path = output_folder  # 文件夹给 base_path
            output.file_slots[0].path = f'{i:03d}_{name}'  # 文件名给 slot
            
        # Render MR
        if arg.save_mr:
            switch_to_mr_render(False, composite_nodes)
            bpy.ops.render.render(write_still=True)
            shutil.copyfile(bpy.context.scene.render.filepath, 
                            bpy.context.scene.render.filepath.replace('.png', '_mr.png'))
            switch_to_color_render(composite_nodes)

        # Render Standard
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()
        
        # === 【关键修改 B】: 文件查找逻辑适配新的路径设置 ===
        final_normal_path = ""
        
        for name, output in outputs.items():
            ext = EXT[output.format.file_format] # exr
            
            # 重新拼接完整路径来查找文件
            # Blender 保存时是: base_path + / + slot_path + 帧号 + .ext
            # 例如: /data/../rendered/000_normal0001.exr
            file_prefix = os.path.join(output.base_path, output.file_slots[0].path)
            pattern = f'{file_prefix}*.{ext}'
            
            found_files = glob.glob(pattern)
            
            if not found_files:
                print(f"[WARN] No output found for {name}. Pattern: {pattern}")
                continue
                
            src_file = found_files[0]
            
            # 目标文件名 (去掉帧号 0001)
            # 注意：这里我们手动拼接 output_folder 和 文件名
            clean_filename = f'{output.file_slots[0].path}.{ext}'
            final_exr_path = os.path.join(output_folder, clean_filename)
            
            # 重命名 (如果文件名含帧号)
            if src_file != final_exr_path:
                # 如果目标文件已存在(可能是之前残留)，先删掉
                if os.path.exists(final_exr_path):
                    os.remove(final_exr_path)
                os.rename(src_file, final_exr_path)
            
            # EXR -> JPG/PNG 转换
            if name == 'normal':
                jpg_filename = f'{output.file_slots[0].path}.jpg'
                jpg_path = os.path.join(output_folder, jpg_filename)
                
                try:
                    ConvertNormalMap(final_exr_path, jpg_path)
                    if os.path.exists(jpg_path):
                        os.remove(final_exr_path) # 删除巨大的EXR
                        final_normal_path = jpg_filename
                    else:
                        print(f"[ERR] Failed to write JPG: {jpg_path}")
                except Exception as e:
                    print(f"[ERR] Normal conversion error: {e}")
                    
            elif name == 'depth':
                png_filename = f'{output.file_slots[0].path}.png'
                png_path = os.path.join(output_folder, png_filename)
                try:
                    ConvertDepthMap(final_exr_path, png_path, cam)
                    if os.path.exists(png_path):
                        os.remove(final_exr_path)
                except Exception as e:
                    print(f"[ERR] Depth conversion error: {e}")

        # Metadata
        metadata = {
            "file_path": rgb_filename,
            "view_index": i,
            "azimuth": view['hangle'],
            "elevation": view['vangle'],
            "camera_angle_x": view['fov'],
            "transform_matrix": get_transform_matrix(cam)
        }
        to_export["frames"].append(metadata)
        
        eval_views_data.append({
            "id": i,
            "azimuth": math.degrees(view['hangle']),
            "elevation": math.degrees(view['vangle']),
            "rgb_path": rgb_filename,
            "normal_path": final_normal_path if final_normal_path else f"{i:03d}_normal.jpg"
        })

    # Save JSONs
    with open(os.path.join(output_folder, 'transforms.json'), 'w') as f:
        json.dump(to_export, f, indent=4)
    with open(os.path.join(output_folder, 'views.json'), 'w') as f:
        json.dump(eval_views_data, f, indent=4)
        
    print(f"Saved outputs for {model_name}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, 
        help='Root directory containing subfolders with models.')
    # output_folder 不再强制需要，因为我们会自动在模型同级目录下创建
    parser.add_argument('--output_folder', type=str, default=None, 
        help='(Optional) Overridden by local "rendered" folder logic.')
    
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--engine', type=str, default='CYCLES')
    parser.add_argument('--save_depth', action='store_true', default=False)
    parser.add_argument('--save_normal', action='store_true', default=True)
    parser.add_argument('--save_albedo', action='store_true', default=False)
    parser.add_argument('--save_mr', action='store_true', default=False)
    parser.add_argument('--save_mist', action='store_true', default=False)
    parser.add_argument('--split_normal', action='store_true', default=False)
    
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
    args = parser.parse_args(argv)

    # 递归查找所有模型文件
    files_to_process = []
    valid_exts = set(IMPORT_FUNCTIONS.keys())
    
    if os.path.isfile(args.input):
        files_to_process.append(args.input)
    else:
        print(f"Scanning directory: {args.input} ...")
        for root, dirs, files in os.walk(args.input):
            # 过滤掉名为 'rendered' 的文件夹，避免重复扫描输出目录
            if 'rendered' in dirs:
                dirs.remove('rendered')
                
            for f in files:
                if f.split('.')[-1].lower() in valid_exts:
                    files_to_process.append(os.path.join(root, f))
    
    print(f"Found {len(files_to_process)} models.")
    
    for model_path in files_to_process:
        try:
            # 1. 获取当前模型所在的目录
            current_model_dir = os.path.dirname(model_path)
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            
            # 2. 构造目标输出根目录： ../子文件夹/rendered
            # process_single_model 会在其下再创建一个 model_name 的文件夹
            local_output_root = os.path.join(current_model_dir, "rendered")
            
            # 3. 检查是否已存在 (Skip Logic)
            # 最终路径会是: .../rendered/{model_name}/views.json
            final_output_dir = os.path.join(local_output_root, model_name)
            if os.path.exists(os.path.join(final_output_dir, "views.json")):
                print(f"[SKIP] Already processed: {model_name} in {current_model_dir}")
                continue

            # 4. 执行处理
            # 注意：我们将 local_output_root 传给函数，函数内部会自动拼接 model_name
            process_single_model(model_path, local_output_root, args)
            
        except Exception as e:
            print(f"[ERROR] Failed processing {model_path}: {e}")