import bpy
import numpy as np
import sys
import os
import traceback

def install_h5py():
    """Install h5py with pip if not available"""
    try:
        import h5py
        return h5py
    except ImportError:
        print("h5py not found, attempting installation...")
        try:
            import subprocess
            python_exe = sys.executable
            subprocess.check_call([python_exe, "-m", "pip", "install", "h5py"])
            import h5py
            return h5py
        except Exception as e:
            print(f"Failed to install h5py: {e}")
            sys.exit(1)

def clear_scene():
    """Clear all objects in the current scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def inspect_h5_structure(filepath):
    """Inspect and print HDF5 file structure"""
    h5py = install_h5py()
    print("\nInspecting HDF5 file structure...")
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"File: {filepath}")
            print("Groups/Datasets:")
            
            def print_attrs(name, obj):
                print(f"  {name}")
                print(f"    Shape: {obj.shape if hasattr(obj, 'shape') else 'N/A'}")
                print(f"    Type: {type(obj)}")
                if isinstance(obj, h5py.Dataset):
                    print(f"    Dataset dtype: {obj.dtype}")
                    print(f"    Dataset chunks: {obj.chunks}")
                    print(f"    Dataset compression: {obj.compression}")
            
            f.visititems(print_attrs)
    except Exception as e:
        print(f"Error inspecting HDF5 file: {e}")

def load_h5_data_safely(filepath):
    """More robust HDF5 data loading with multiple fallback methods"""
    h5py = install_h5py()
    
    try:
        # First try the standard approach
        with h5py.File(filepath, 'r') as f:
            # Try to find a suitable dataset
            for name in f.keys():
                dataset = f[name]
                if isinstance(dataset, h5py.Dataset):
                    data = np.array(dataset)
                    
                    # Check if data looks like point cloud
                    if len(data.shape) == 2 and data.shape[1] >= 3:
                        return data[:, :3]  # Return first 3 columns as XYZ
            
            # If no suitable dataset found, try alternative approaches
            print("No standard dataset found, trying alternative methods...")
            
            # Method 1: Try to load all data concatenated
            all_data = []
            for name in f.keys():
                dataset = f[name]
                if isinstance(dataset, h5py.Dataset):
                    all_data.append(np.array(dataset))
            
            if all_data:
                combined = np.concatenate(all_data)
                if len(combined.shape) == 2 and combined.shape[1] >= 3:
                    return combined[:, :3]
            
            # Method 2: Try to load attributes
            if 'vertices' in f.attrs:
                vertices = np.array(f.attrs['vertices'])
                if len(vertices.shape) == 2 and vertices.shape[1] >= 3:
                    return vertices[:, :3]
            
            # If nothing worked, raise error
            raise ValueError("No suitable 3D point data found in HDF5 file")
            
    except Exception as e:
        print(f"Error in load_h5_data_safely: {e}")
        inspect_h5_structure(filepath)
        raise

def create_mesh_from_points(vertices, name="H5Mesh"):
    """Create a mesh from point data with better visualization"""
    # Create mesh and object
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    
    # Link object to scene
    bpy.context.collection.objects.link(obj)
    
    # Create mesh from vertices (as point cloud)
    mesh.from_pydata(vertices, [], [])
    mesh.update()
    
    # Add material for better visualization
    mat = bpy.data.materials.new(name="PointCloudMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # Create simple principled shader
    shader = nodes.new(type='ShaderNodeBsdfPrincipled')
    shader.inputs['Base Color'].default_value = (0.8, 0.2, 0.1, 1)
    shader.inputs['Metallic'].default_value = 0.0
    shader.inputs['Roughness'].default_value = 0.5
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(shader.outputs['BSDF'], output.inputs['Surface'])
    
    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    return obj

def setup_camera_and_light(target_obj):
    """Improved camera and lighting setup"""
    # Create camera
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "RenderCamera"
    
    # Calculate bounding box to position camera appropriately
    bbox = [target_obj.matrix_world @ v.co for v in target_obj.bound_box]
    bbox_center = sum(bbox, bpy.math.Vector()) / 8
    bbox_size = max((max(v[i] for v in bbox) - min(v[i] for v in bbox)) for i in range(3))
    
    # Position camera
    camera_distance = bbox_size * 2.5
    camera.location = bbox_center + bpy.math.Vector((camera_distance, -camera_distance, camera_distance))
    
    # Point camera to object center
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = target_obj
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    # Set this camera as active
    bpy.context.scene.camera = camera
    
    # Add lighting
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 10))
    sun1 = bpy.context.object
    sun1.data.energy = 2.0
    
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, -10))
    sun2 = bpy.context.object
    sun2.data.energy = 1.0
    sun2.rotation_euler = (3.14159, 0, 0)  # Point upwards
    
    # Add area light for fill
    bpy.ops.object.light_add(type='AREA', radius=5, location=(10, 0, 0))
    area = bpy.context.object
    area.data.energy = 300
    area.rotation_euler = (0, 1.5708, 0)  # Point towards center
    
    # Set render engine to Cycles for better quality
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64  # Reduce noise

def render_to_png(output_path):
    """Render scene to PNG with better settings"""
    # Set output settings
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.image_settings.color_depth = '8'
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.filepath = output_path
    
    # Render
    print("Starting render...")
    bpy.ops.render.render(write_still=True)
    print(f"Render complete: {output_path}")

def main(h5_filepath, output_png_path):
    """Main function to process H5 file and render to PNG"""
    try:
        print("\nStarting HDF5 to PNG conversion...")
        
        # First inspect the HDF5 file structure
        inspect_h5_structure(h5_filepath)
        
        # Clear existing scene
        clear_scene()
        
        # Load data from H5 file
        print("\nLoading data from HDF5 file...")
        vertices = load_h5_data_safely(h5_filepath)
        print(f"Successfully loaded {len(vertices)} points")
        
        # Create mesh from points
        print("Creating mesh...")
        obj = create_mesh_from_points(vertices)
        
        # Set up camera and lighting
        print("Setting up camera and lights...")
        setup_camera_and_light(obj)
        
        # Render to PNG
        print("Starting rendering process...")
        render_to_png(output_png_path)
        
        print("\nProcess completed successfully!")
        
    except Exception as e:
        print("\nERROR during processing:")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # Set your input and output paths here
    h5_filepath = "/home/devel/.draft/renderformer/tmp/cbox/cbox.h5"  # Change this to your H5 file path
    output_png_path = "render.png"  # Change this to your desired output path
    
    # Run the main function
    main(h5_filepath, output_png_path)