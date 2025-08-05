import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Union, Optional, List
import numpy as np

class BlenderRenderConfig:
    """Render configuration"""
    def __init__(
        self,
        resolution: int = 512,
        samples: int = 128,
        use_gpu: bool = True,
        use_denoising: bool = True,
        denoiser_type: str = 'OPTIX',
        transparent: bool = True,
        light_bounces: int = 8,
        caustics: bool = True,
        exposure: float = 1.0,
        film_transparent: bool = True,
        color_mode: str = 'RGBA',
        color_depth: str = '32',
        exr_codec: str = 'DWAA',
        use_motion_blur: bool = False,
        use_dof: bool = False,
        use_bloom: bool = False
    ):
        self.resolution = resolution
        self.samples = samples
        self.use_gpu = use_gpu
        self.use_denoising = use_denoising
        self.denoiser_type = denoiser_type
        self.transparent = transparent
        self.light_bounces = light_bounces
        self.caustics = caustics
        self.exposure = exposure
        self.film_transparent = film_transparent
        self.color_mode = color_mode
        self.color_depth = color_depth
        self.exr_codec = exr_codec
        self.use_motion_blur = use_motion_blur
        self.use_dof = use_dof
        self.use_bloom = use_bloom

class BlenderRenderer:
    def __init__(self, config: Optional[BlenderRenderConfig] = None):
        self.config = config or BlenderRenderConfig()
        self._check_blender_installation()

    def _check_blender_installation(self):
        """Check Blender installation"""
        try:
            result = subprocess.run(['blender', '--version'], 
                                 capture_output=True, 
                                 text=True, 
                                 check=True)
            print(f"Found Blender: {result.stdout.splitlines()[0]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "Blender is not installed. Install with: sudo apt install blender"
            )

    def _generate_material_setup(self) -> str:
        """Generate material setup code"""
        return """
def setup_material(material_data, name="Material"):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    
    # Create base nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    
    # Configure Principled BSDF
    if 'diffuse' in material_data:
        principled.inputs['Base Color'].default_value = (*material_data['diffuse'], 1.0)
    if 'specular' in material_data:
        principled.inputs['Specular'].default_value = sum(material_data['specular']) / 3
    if 'roughness' in material_data:
        principled.inputs['Roughness'].default_value = material_data['roughness']
    if 'metallic' in material_data:
        principled.inputs['Metallic'].default_value = material_data['metallic']
    
    # Add emission if present
    if 'emissive' in material_data and sum(material_data['emissive']) > 0:
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs['Color'].default_value = (*material_data['emissive'], 1.0)
        emission_strength = sum(material_data['emissive']) * 3
        emission.inputs['Strength'].default_value = emission_strength
        
        # Mix with principled
        mix = nodes.new('ShaderNodeMixShader')
        mix.inputs[0].default_value = min(emission_strength, 1.0)
        links.new(principled.outputs['BSDF'], mix.inputs[1])
        links.new(emission.outputs['Emission'], mix.inputs[2])
        links.new(mix.outputs['Shader'], output.inputs['Surface'])
    else:
        # Direct connection if no emission
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    return material
"""

    def _generate_scene_setup(self) -> str:
        """Generate scene setup code"""
        return f"""
import bpy
import numpy as np

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Render settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.resolution_x = {self.config.resolution}
bpy.context.scene.render.resolution_y = {self.config.resolution}
bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.render.film_transparent = {str(self.config.film_transparent).lower()}

# Cycles settings
cycles = bpy.context.scene.cycles
cycles.samples = {self.config.samples}
cycles.max_bounces = {self.config.light_bounces}
cycles.caustics_reflective = {str(self.config.caustics).lower()}
cycles.caustics_refractive = {str(self.config.caustics).lower()}
cycles.use_denoising = {str(self.config.use_denoising).lower()}
cycles.denoiser = '{self.config.denoiser_type}'

# Try to use GPU
if {str(self.config.use_gpu).lower()}:
    try:
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            device.use = True
        cycles.device = 'GPU'
        print("Using GPU for rendering")
    except Exception as e:
        print(f"GPU not available, falling back to CPU. Error: {{str(e)}}")
        cycles.device = 'CPU'

# Output settings
bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
bpy.context.scene.render.image_settings.color_mode = '{self.config.color_mode}'
bpy.context.scene.render.image_settings.color_depth = '{self.config.color_depth}'
bpy.context.scene.render.image_settings.exr_codec = '{self.config.exr_codec}'

# Additional effects
bpy.context.scene.render.use_motion_blur = {str(self.config.use_motion_blur).lower()}
bpy.context.scene.render.use_bloom = {str(self.config.use_bloom).lower()}
"""

    def _generate_camera_setup(self) -> str:
        """Generate camera setup code"""
        return """
def setup_camera(camera_data):
    # Create camera
    cam_data = bpy.data.cameras.new('Camera')
    cam = bpy.data.objects.new('Camera', cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    
    # Position and orientation
    cam.location = camera_data.get('position', (0, 0, 0))
    
    if 'look_at' in camera_data:
        # Point camera at target
        direction = (
            camera_data['look_at'][0] - cam.location.x,
            camera_data['look_at'][1] - cam.location.y,
            camera_data['look_at'][2] - cam.location.z
        )
        rot_quat = Vector(direction).to_track_quat('-Z', 'Y')
        cam.rotation_euler = rot_quat.to_euler()
    else:
        # Use rotation if look_at not specified
        cam.rotation_euler = camera_data.get('rotation', (0, 0, 0))
    
    # Camera settings
    cam_data.lens_unit = 'FOV'
    cam_data.angle = np.radians(camera_data.get('fov', 45))
    cam_data.dof.use_dof = {str(self.config.use_dof).lower()}
    
    return cam
"""

    def _generate_mesh_setup(self) -> str:
        """Generate mesh creation code"""
        return """
def create_mesh(obj_name, vertices, faces, material=None, transform=None):
    # Create mesh
    mesh = bpy.data.meshes.new(obj_name)
    obj = bpy.data.objects.new(obj_name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    
    # Create geometry
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    
    # Apply material
    if material:
        obj.data.materials.append(material)
    
    # Apply transformations
    if transform:
        if 'translation' in transform:
            obj.location = transform['translation']
        if 'rotation' in transform:
            obj.rotation_euler = [np.radians(r) for r in transform['rotation']]
        if 'scale' in transform:
            obj.scale = transform['scale']
    
    return obj
"""

    def _create_blender_script(self, scene_data: Dict, output_path: str) -> str:
        """Create complete Blender script"""
        script = self._generate_scene_setup()
        script += """
from mathutils import Vector
"""
        script += self._generate_material_setup()
        script += self._generate_camera_setup()
        script += self._generate_mesh_setup()
        
        script += f"""
# Scene data
scene_data = {json.dumps(scene_data, indent=4)}

# Create materials first
materials = {{}}
for obj_name, obj_data in scene_data.get('objects', {{}}).items():
    if 'material' in obj_data:
        materials[obj_name] = setup_material(obj_data['material'], obj_name)

# Create objects
for obj_name, obj_data in scene_data.get('objects', {{}}).items():
    # Get vertices and faces
    vertices = np.array(obj_data.get('vertices', []))
    faces = np.array(obj_data.get('faces', []))
    
    if len(vertices) == 0 or len(faces) == 0:
        print(f"Skipping object {{obj_name}} - no geometry data")
        continue
    
    # Create object
    obj = create_mesh(
        obj_name,
        vertices,
        faces,
        materials.get(obj_name),
        obj_data.get('transform')
    )

# Setup camera if available
if 'cameras' in scene_data and len(scene_data['cameras']) > 0:
    setup_camera(scene_data['cameras'][0])
else:
    # Default camera if none provided
    print("No camera data found, creating default camera")
    cam_data = bpy.data.cameras.new('Camera')
    cam = bpy.data.objects.new('Camera', cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.location = (5, -5, 3)
    cam.rotation_euler = (np.radians(60), 0, np.radians(45))

# Render HDR
bpy.context.scene.render.filepath = '{output_path}'
bpy.ops.render.render(write_still=True)

# Render LDR (PNG)
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.color_mode = 'RGBA' if {str(self.config.transparent).lower()} else 'RGB'
bpy.context.scene.render.image_settings.color_depth = '8'
bpy.context.scene.render.filepath = '{output_path}'.replace('.exr', '.png')
bpy.ops.render.render(write_still=True)
"""
        return script

    def render_scene(self, scene_data: Dict, output_path: Union[str, Path]):
        """Render scene using Blender"""
        output_path = str(output_path)
        if not output_path.endswith('.exr'):
            output_path = os.path.splitext(output_path)[0] + '.exr'
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create temporary script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            script_path = f.name
            f.write(self._create_blender_script(scene_data, output_path))
        
        try:
            # Run Blender
            process = subprocess.Popen(
                [
                    'blender',
                    '--background',
                    '--python', script_path,
                    '--enable-autoexec'
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Print progress
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # Check result
            if process.returncode != 0:
                error_output = process.stderr.read()
                raise RuntimeError(f"Blender failed with error: {{error_output}}")
            
            print(f"Rendering completed successfully:")
            print(f"  - HDR: {output_path}")
            print(f"  - LDR: {output_path.replace('.exr', '.png')}")
            
        finally:
            # Clean up temporary script
            if os.path.exists(script_path):
                os.remove(script_path)