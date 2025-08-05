import json
import random
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

CONFIG = {
    "OUT_PATH": "/home/devel/.draft/renderformer/datasets"
}


class SceneGenerator:
    def __init__(self, base_path: str = "/home/devel/.draft/renderformer/examples"):
        self.base_path = Path(base_path)
        self.templates_path = self.base_path / "templates"
        self.objects_path = self.base_path / "objects"
        
    def generate_cornell_box_scene(
        self,
        scene_name: str,
        object_name: str = "bunny",
        object_path: str = "classical/bunny.obj",
        object_scale: float = 1.0,
        object_material: Optional[Dict] = None,
        light_intensity: float = 5000.0,
        light_scale: float = 2.5,
        wall_colors: Optional[List[List[float]]] = None,
        camera_position: List[float] = [0.0, -2.0, 0.0],
        camera_look_at: List[float] = [0.0, 0.0, 0.0],
        camera_fov: float = 37.5,
        version: str = "1.0"
    ) -> Dict:
        """Generate a Cornell box scene with a specified object."""
        
        # Default wall colors if not provided
        if wall_colors is None:
            wall_colors = [
                [0.4, 0.4, 0.4],  # floor
                [0.4, 0.4, 0.4],  # back wall
                [0.0, 0.5, 0.0],  # right wall (green)
                [0.5, 0.0, 0.0]   # left wall (red)
            ]
            
        # Default object material if not provided
        if object_material is None:
            object_material = {
                "diffuse": [0.0, 0.0, 0.0],
                "specular": [0.9, 0.9, 0.9],
                "random_diffuse_max": 0.1,
                "roughness": 0.1,
                "emissive": [0.0, 0.0, 0.0],
                "smooth_shading": True,
                "rand_tri_diffuse_seed": None
            }
        
        scene = {
            "scene_name": scene_name,
            "version": version,
            "objects": {
                "background_0": {
                    "mesh_path": str(self.templates_path / "backgrounds/plane.obj"),
                    "transform": {
                        "translation": [0.0, 0.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0],
                        "scale": [0.5, 0.5, 0.5],
                        "normalize": False
                    },
                    "material": {
                        "diffuse": wall_colors[0],
                        "specular": [0.01, 0.01, 0.01],
                        "random_diffuse_max": 0.4,
                        "roughness": 0.99,
                        "emissive": [0.0, 0.0, 0.0],
                        "smooth_shading": True,
                        "rand_tri_diffuse_seed": None
                    }
                },
                "background_1": {
                    "mesh_path": str(self.templates_path / "backgrounds/wall0.obj"),
                    "transform": {
                        "translation": [0.0, 0.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0],
                        "scale": [0.5, 0.5, 0.5],
                        "normalize": False
                    },
                    "material": {
                        "diffuse": wall_colors[1],
                        "specular": [0.55, 0.55, 0.55],
                        "random_diffuse_max": 0.4,
                        "roughness": 0.1,
                        "emissive": [0.0, 0.0, 0.0],
                        "smooth_shading": True,
                        "rand_tri_diffuse_seed": None
                    }
                },
                "background_2": {
                    "mesh_path": str(self.templates_path / "backgrounds/wall1.obj"),
                    "transform": {
                        "translation": [0.0, 0.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0],
                        "scale": [0.5, 0.5, 0.5],
                        "normalize": False
                    },
                    "material": {
                        "diffuse": wall_colors[2],
                        "specular": [0.01, 0.01, 0.01],
                        "random_diffuse_max": 0.5,
                        "roughness": 0.99,
                        "emissive": [0.0, 0.0, 0.0],
                        "smooth_shading": True,
                        "rand_tri_diffuse_seed": None
                    }
                },
                "background_3": {
                    "mesh_path": str(self.templates_path / "backgrounds/wall2.obj"),
                    "transform": {
                        "translation": [0.0, 0.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0],
                        "scale": [0.5, 0.5, 0.5],
                        "normalize": False
                    },
                    "material": {
                        "diffuse": wall_colors[3],
                        "specular": [0.01, 0.01, 0.01],
                        "random_diffuse_max": 0.5,
                        "roughness": 0.99,
                        "emissive": [0.0, 0.0, 0.0],
                        "smooth_shading": False,
                        "rand_tri_diffuse_seed": None
                    }
                },
                object_name: {
                    "mesh_path": str(self.objects_path / object_path),
                    "transform": {
                        "translation": [0.0, 0.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0],
                        "scale": [object_scale, object_scale, object_scale],
                        "normalize": True
                    },
                    "material": object_material
                },
                "light_0": {
                    "mesh_path": str(self.templates_path / "lighting/tri.obj"),
                    "transform": {
                        "translation": [0.0, 0.0, 2.1],
                        "rotation": [0.0, 0.0, 0.0],
                        "scale": [light_scale, light_scale, light_scale],
                        "normalize": False
                    },
                    "material": {
                        "diffuse": [1.0, 1.0, 1.0],
                        "specular": [0.0, 0.0, 0.0],
                        "random_diffuse_max": 0.0,
                        "roughness": 1.0,
                        "emissive": [light_intensity, light_intensity, light_intensity],
                        "smooth_shading": False,
                        "rand_tri_diffuse_seed": None
                    }
                }
            },
            "cameras": [
                {
                    "position": camera_position,
                    "look_at": camera_look_at,
                    "up": [0.0, 0.0, 1.0],
                    "fov": camera_fov
                }
            ]
        }
        
        return scene
    
    def save_scene(self, scene: Dict, output_path: Union[str, Path]):
        """Save the scene configuration to a JSON file."""
        with open(output_path, 'w') as f:
            json.dump(scene, f, indent=4)
    
    def generate_random_cornell_box(self):
        """Generate a Cornell box scene with random parameters."""
        objects = [
            ("bunny", "classical/bunny.obj"),
            ("lucy", "lucy/lucy.obj"),
            ("teapot", "classical/teapot.obj"),
            ("short-box", "cbox/short-box.obj"),
            ("tall-box", "cbox/tall-box.obj")
        ]
        
        object_name, object_path = random.choice(objects)
        
        # Random material properties
        material = {
            "diffuse": [random.random() * 0.3, random.random() * 0.3, random.random() * 0.3],
            "specular": [random.random(), random.random(), random.random()],
            "random_diffuse_max": random.random() * 0.3,
            "roughness": random.random(),
            "emissive": [0.0, 0.0, 0.0],
            "smooth_shading": random.choice([True, False]),
            "rand_tri_diffuse_seed": None
        }
        
        # Random wall colors
        wall_colors = [
            [0.4, 0.4, 0.4],  # floor (keep neutral)
            [0.4, 0.4, 0.4],  # back wall (keep neutral)
            [random.random() * 0.5, random.random() * 0.5 + 0.5, random.random() * 0.5],  # right wall
            [random.random() * 0.5 + 0.5, random.random() * 0.5, random.random() * 0.5]   # left wall
        ]
        
        scene = self.generate_cornell_box_scene(
            scene_name=f"cornell_box_{object_name}",
            object_name=object_name,
            object_path=object_path,
            object_scale=random.uniform(0.8, 1.2),
            object_material=material,
            light_intensity=random.uniform(3000.0, 7000.0),
            light_scale=random.uniform(2.0, 3.0),
            wall_colors=wall_colors,
            camera_position=[
                random.uniform(-1.0, 1.0),
                random.uniform(-3.0, -1.5),
                random.uniform(0.0, 1.0)
            ],
            camera_fov=random.uniform(30.0, 45.0)
        )
        
        return scene

if __name__ == "__main__":
    generator = SceneGenerator()
    
    # Generate standard scenes
    scenes = [
        ("cornell_box_bunny", "bunny", "classical/bunny.obj"),
        ("cornell_box_teapot", "teapot", "classical/teapot.obj"),
        ("cornell_box_lucy", "lucy", "lucy/lucy.obj"),
        ("cornell_box", None, None)  # Empty cornell box
    ]
    
    for scene_name, obj_name, obj_path in scenes:
        if obj_name is None:
            # Empty cornell box
            scene = generator.generate_cornell_box_scene(
                scene_name=scene_name,
                object_name="short-box",
                object_path="cbox/short-box.obj",
                object_scale=1.0
            )
        else:
            scene = generator.generate_cornell_box_scene(
                scene_name=scene_name,
                object_name=obj_name,
                object_path=obj_path
            )
        
        output_path = f"{CONFIG['OUT_PATH']}/{scene_name}.json"
        generator.save_scene(scene, output_path)
        print(f"Generated {output_path}")
    
    # Generate some random variations
    for i in range(5):
        scene = generator.generate_random_cornell_box()
        output_path = f"{CONFIG['OUT_PATH']}/random_cornell_box_{i}.json"
        generator.save_scene(scene, output_path)
        print(f"Generated {output_path}")