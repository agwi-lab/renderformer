import blenderproc as bproc
import json
import argparse
import os
import numpy as np
from pathlib import Path
from PIL import Image


def render_scene_from_json(json_path: str, save_dir: str, image_name: str):

    print("start rendering by custom function")

    # Load JSON configuration
    with open(json_path, 'r') as f:
        scene_config = json.load(f)

    bproc.init()

    # Clear the scene first
    bproc.clean_up()

    # Load all objects from the scene configuration
    for obj_name, obj_data in scene_config["objects"].items():
        # Get the mesh path and adjust it to absolute path
        mesh_path = obj_data["mesh_path"]

        # Load the object
        obj = bproc.loader.load_obj(mesh_path)[0]
        
        # Apply transformation
        transform = obj_data["transform"]
        obj.set_location(transform["translation"])
        obj.set_rotation_euler([np.radians(angle) for angle in transform["rotation"]])
        obj.set_scale(transform["scale"])
        
        # Apply material properties
        material = obj_data["material"]
        mat = obj.get_materials()[0] if obj.get_materials() else bproc.material.create("Material")

        # Convert RGB to RGBA (add alpha channel)
        diffuse_rgba = material["diffuse"] + [1.0]  # Add alpha=1.0
        
        # Set material properties
        mat.set_principled_shader_value("Base Color", diffuse_rgba)
        mat.set_principled_shader_value("Metallic", 0.0)  # Non-metallic
        mat.set_principled_shader_value("Roughness", material["roughness"])
        
        # Handle specular - use average of specular components for Specular IOR Level
        specular_avg = sum(material["specular"]) / 3.0
        mat.set_principled_shader_value("Specular IOR Level", specular_avg)

        # Handle emissive materials (lights)
        if any(emissive > 0 for emissive in material["emissive"]):
            # Convert emissive to RGBA and set emission
            emissive_avg = sum(material["emissive"]) / 3
            mat.set_principled_shader_value("Emission Strength", emissive_avg)

        if not obj.get_materials():
            obj.add_material(mat)

    # Set up cameras
    for camera_config in scene_config["cameras"]:
        position = camera_config["position"]
        look_at = camera_config["look_at"]
        up = camera_config["up"]
        fov_degrees = camera_config["fov"]

        # Calculate rotation to look at the target point
        direction = np.array(look_at) - np.array(position)
        rotation = bproc.camera.rotation_from_forward_vec(direction)
        
        # Create camera matrix with correct rotation
        cam_pose = bproc.math.build_transformation_mat(position, rotation)
        bproc.camera.add_camera_pose(cam_pose)

        # Set camera properties - convert FOV from degrees to radians for the lens parameter
        fov_radians = np.radians(fov_degrees)
        bproc.camera.set_intrinsics_from_blender_params(
            lens=fov_radians,  # Use 'lens' parameter with FOV in radians
            lens_unit="FOV"
        )

    # Render the scene
    data = bproc.renderer.render()

    # Get the rendered data
    colors = data['colors']
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the image
    image_path = os.path.join(save_dir, image_name)
    print(f"Saving image to {image_path}")
    image = Image.fromarray((colors[0] * 255).astype(np.uint8))
    image.save(image_path)

def main():
    parser = argparse.ArgumentParser(description="Render a scene from JSON using BlenderProc")
    parser.add_argument("--json_path", "-j", help="Path to the scene JSON file")
    parser.add_argument("--output_path", "-o", help="Path to the scene output file")
    parser.add_argument("--image_name", "-i", help="Name of the scene image file")

    args = parser.parse_args()

    json_path = args.json_path
    output_path = args.output_path
    image_name = args.image_name

    render_scene_from_json(json_path, output_path, image_name)

if __name__ == "__main__":
    main()
