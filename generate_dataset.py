import json
import random
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import glob
from dacite import from_dict, Config
import sys
import numpy as np
import h5py

# Добавляем путь к scene_processor в PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), "scene_processor"))

from scene_processor.to_h5 import save_to_h5
from scene_processor.scene_mesh import generate_scene_mesh
from scene_processor.scene_config import SceneConfig
from scene_processor.blender_render import BlenderRenderer, BlenderRenderConfig

CONFIG = {
    "DATA_PATH": "/home/devel/.draft/renderformer/datasets",
    "JSON_PATH": "/home/devel/.draft/renderformer/datasets/json",
    "H5_PATH": "/home/devel/.draft/renderformer/datasets/h5",
    "GT_PATH": "/home/devel/.draft/renderformer/datasets/gt",
    "TEMP_MESH_PATH": "/home/devel/.draft/renderformer/datasets/temp",
    "OBJ_PATH": "/home/devel/.draft/renderformer/examples/objects",
    "TMP_PATH": "/home/devel/.draft/renderformer/examples/templates",
    "NUM_RANDOM_SCENES": 20,
    "RENDER": {
        "resolution": 512,
        "samples": 256,
        "use_gpu": True,
        "use_denoising": True,
        "light_bounces": 8,
        "caustics": True,
        "exposure": 1.0,
        "use_bloom": True
    }
}

class SceneGenerator:
    def __init__(self):
        self.templates_path = Path(CONFIG["TMP_PATH"])
        self.objects_path = Path(CONFIG["OBJ_PATH"])
        self.json_path = Path(CONFIG["JSON_PATH"])
        self.h5_path = Path(CONFIG["H5_PATH"])
        self.gt_path = Path(CONFIG["GT_PATH"])
        self.temp_mesh_path = Path(CONFIG["TEMP_MESH_PATH"])
        
        # Создаем необходимые директории
        for path in [self.json_path, self.h5_path, self.gt_path, self.temp_mesh_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Собираем все доступные объекты
        self.available_objects = self._collect_objects()
        
        # Инициализируем рендерер
        render_config = BlenderRenderConfig(**CONFIG["RENDER"])
        self.renderer = BlenderRenderer(config=render_config)

    def _collect_objects(self) -> List[tuple]:
        """Собирает все доступные .obj файлы из директории objects"""
        objects = []
        for obj_file in glob.glob(str(self.objects_path / "**/*.obj"), recursive=True):
            rel_path = os.path.relpath(obj_file, str(self.objects_path))
            obj_name = Path(rel_path).stem
            objects.append((obj_name, rel_path))
        return objects

    def prepare_scene_for_blender(self, scene: Dict) -> Dict:
        """Подготавливает сцену для рендеринга в Blender"""
        blender_scene = {
            "objects": {},
            "cameras": scene["cameras"]
        }

        project_root = Path(__file__).parent

        # Преобразуем пути к абсолютным и загружаем геометрию
        for obj_name, obj_data in scene["objects"].items():
            mesh_path = project_root / obj_data["mesh_path"]
            
            # Загружаем меш для получения вершин и граней
            with open(mesh_path, 'r') as f:
                vertices = []
                faces = []
                for line in f:
                    if line.startswith('v '):
                        v = [float(x) for x in line.split()[1:4]]
                        vertices.append(v)
                    elif line.startswith('f '):
                        f = [int(x.split('/')[0]) - 1 for x in line.split()[1:4]]
                        faces.append(f)

            blender_scene["objects"][obj_name] = {
                "vertices": vertices,
                "faces": faces,
                "transform": obj_data["transform"],
                "material": obj_data["material"]
            }

        return blender_scene

    def generate_scene(self, scene_name: str, object_name: str, object_path: str, random_params: bool = True) -> Dict:
        """Генерирует сцену с заданным объектом"""
        
        # Случайные параметры для сцены
        if random_params:
            wall_colors = [
                [random.uniform(0.2, 0.6)] * 3,  # floor
                [random.uniform(0.2, 0.6)] * 3,  # back wall
                [random.uniform(0.0, 0.7), random.uniform(0.3, 0.8), random.uniform(0.0, 0.7)],  # right wall
                [random.uniform(0.3, 0.8), random.uniform(0.0, 0.7), random.uniform(0.0, 0.7)]   # left wall
            ]
            
            camera_position = [
                random.uniform(-2.0, 2.0),
                random.uniform(-3.0, -1.0),
                random.uniform(-0.5, 1.5)
            ]
            
            light_params = {
                "intensity": random.uniform(3000.0, 8000.0),
                "scale": random.uniform(1.5, 3.5),
                "position": [
                    random.uniform(-1.0, 1.0),
                    random.uniform(-1.0, 1.0),
                    random.uniform(1.8, 2.3)
                ]
            }
            
            object_params = {
                "scale": random.uniform(0.5, 1.5),
                "rotation": [
                    random.uniform(0, 360),
                    random.uniform(0, 360),
                    random.uniform(0, 360)
                ],
                "position": [
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.5, 0.5)
                ]
            }
        else:
            wall_colors = [
                [0.4, 0.4, 0.4],  # floor
                [0.4, 0.4, 0.4],  # back wall
                [0.0, 0.5, 0.0],  # right wall (green)
                [0.5, 0.0, 0.0]   # left wall (red)
            ]
            camera_position = [0.0, -2.0, 0.0]
            light_params = {
                "intensity": 5000.0,
                "scale": 2.5,
                "position": [0.0, 0.0, 2.1]
            }
            object_params = {
                "scale": 1.0,
                "rotation": [0.0, 0.0, 0.0],
                "position": [0.0, 0.0, 0.0]
            }

        scene = {
            "scene_name": scene_name,
            "version": "1.0",
            "objects": {
                "background_0": {
                    "mesh_path": "examples/templates/backgrounds/plane.obj",
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
                        "rand_tri_diffuse_seed": None,
                        "random_diffuse_type": "per-triangle"
                    },
                    "remesh": False,
                    "remesh_target_face_num": 2048
                },
                # ... (аналогично для остальных стен) ...
                object_name: {
                    "mesh_path": f"examples/objects/{object_path}",
                    "transform": {
                        "translation": object_params["position"],
                        "rotation": object_params["rotation"],
                        "scale": [object_params["scale"]] * 3,
                        "normalize": True
                    },
                    "material": {
                        "diffuse": [random.uniform(0.0, 0.8) for _ in range(3)] if random_params else [0.5, 0.5, 0.5],
                        "specular": [random.uniform(0.7, 1.0)] * 3 if random_params else [0.9, 0.9, 0.9],
                        "random_diffuse_max": random.uniform(0.0, 0.2) if random_params else 0.1,
                        "roughness": random.uniform(0.0, 0.3) if random_params else 0.1,
                        "emissive": [0.0, 0.0, 0.0],
                        "smooth_shading": True,
                        "rand_tri_diffuse_seed": None,
                        "random_diffuse_type": "per-triangle"
                    },
                    "remesh": False,
                    "remesh_target_face_num": 2048
                },
                "light_0": {
                    "mesh_path": "examples/templates/lighting/tri.obj",
                    "transform": {
                        "translation": light_params["position"],
                        "rotation": [0.0, 0.0, 0.0],
                        "scale": [light_params["scale"]] * 3,
                        "normalize": False
                    },
                    "material": {
                        "diffuse": [1.0, 1.0, 1.0],
                        "specular": [0.0, 0.0, 0.0],
                        "random_diffuse_max": 0.0,
                        "roughness": 1.0,
                        "emissive": [light_params["intensity"]] * 3,
                        "smooth_shading": False,
                        "rand_tri_diffuse_seed": None,
                        "random_diffuse_type": "per-triangle"
                    },
                    "remesh": False,
                    "remesh_target_face_num": 2048
                }
            },
            "cameras": [
                {
                    "position": camera_position,
                    "look_at": [0.0, 0.0, 0.0],
                    "up": [0.0, 0.0, 1.0],
                    "fov": random.uniform(30.0, 45.0) if random_params else 37.5
                }
            ]
        }
        
        return scene

    def save_scene(self, scene: Dict, scene_name: str):
        """Сохраняет сцену в JSON, рендерит через Blender и конвертирует в H5"""
        try:
            # Сохраняем JSON
            json_path = self.json_path / f"{scene_name}.json"
            with open(json_path, 'w') as f:
                json.dump(scene, f, indent=4)
            print(f"Saved JSON: {json_path}")

            # Подготавливаем сцену для Blender
            blender_scene = self.prepare_scene_for_blender(scene)
            
            # Рендерим через Blender
            gt_path = self.gt_path / f"{scene_name}.exr"
            self.renderer.render_scene(blender_scene, gt_path)
            print(f"Rendered scene to: {gt_path}")

            # Создаем конфиг сцены для H5
            scene_config = from_dict(
                data_class=SceneConfig, 
                data=scene, 
                config=Config(check_types=True, strict=True)
            )
            
            # Генерируем временный меш
            temp_mesh_file = self.temp_mesh_path / f"{scene_name}.obj"
            project_root = Path(__file__).parent
            
            # Генерируем меш сцены
            generate_scene_mesh(
                scene_config, 
                str(temp_mesh_file),
                str(project_root)
            )
            print(f"Generated mesh: {temp_mesh_file}")
            
            # Сохраняем H5
            h5_path = self.h5_path / f"{scene_name}.h5"
            save_to_h5(scene_config, str(temp_mesh_file), str(h5_path))
            print(f"Saved H5: {h5_path}")
            
            # Удаляем временные файлы
            if temp_mesh_file.exists():
                temp_mesh_file.unlink()
            
            print(f"\nSuccessfully processed scene {scene_name}:")
            print(f"  - JSON: {json_path}")
            print(f"  - Ground Truth: {gt_path}")
            print(f"  - H5: {h5_path}")
            
        except Exception as e:
            print(f"\nError processing scene {scene_name}: {str(e)}")
            print(f"JSON file saved at: {json_path}")
            import traceback
            print("\nDetailed error:")
            print(traceback.format_exc())

def process_scene_batch(generator: SceneGenerator, scenes: List[tuple], random_params: bool = False):
    """Обрабатывает группу сцен с выводом прогресса"""
    total = len(scenes)
    for i, (obj_name, obj_path) in enumerate(scenes, 1):
        print(f"\nProcessing scene {i}/{total}: {obj_name}")
        scene = generator.generate_scene(
            scene_name=f"{'random' if random_params else 'base'}_scene_{obj_name}",
            object_name=obj_name,
            object_path=obj_path,
            random_params=random_params
        )
        generator.save_scene(scene, f"{'random' if random_params else 'base'}_scene_{obj_name}")

def main():
    print("Initializing scene generator...")
    generator = SceneGenerator()
    
    print(f"\nFound {len(generator.available_objects)} objects")
    print("Starting dataset generation...")
    
    # Создаем базовые сцены
    print("\nGenerating base scenes...")
    process_scene_batch(generator, generator.available_objects, random_params=False)
    
    # Генерируем случайные сцены
    print("\nGenerating random scenes...")
    random_scenes = [(obj_name, obj_path) 
                    for _ in range(CONFIG["NUM_RANDOM_SCENES"])
                    for obj_name, obj_path in [random.choice(generator.available_objects)]]
    process_scene_batch(generator, random_scenes, random_params=True)
    
    print("\nDataset generation completed!")

if __name__ == "__main__":
    main()