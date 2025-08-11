import json
import random
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import glob
from dacite import from_dict, Config
import sys

from sympy.parsing.sympy_parser import null

from scene_processor.to_h5 import save_to_h5
from scene_processor.scene_mesh import generate_scene_mesh
from scene_processor.scene_config import SceneConfig
# from scene_processor.render_scene import render_scene_from_json


CONFIG = {
    "DATA_PATH": "/home/devel/.draft/renderformer/datasets",
    "JSON_PATH": "/home/devel/.draft/renderformer/datasets/json",
    "H5_PATH": "/home/devel/.draft/renderformer/datasets/h5",
    "GT_PATH": "/home/devel/.draft/renderformer/datasets/gt",
    "TEMP_MESH_PATH": "/home/devel/.draft/renderformer/datasets/temp",
    "OBJ_PATH": "/home/devel/.draft/renderformer/examples/objects",
    "TMP_PATH": "/home/devel/.draft/renderformer/examples/templates",
    "BASE_DIR": "examples",
    "NUM_RANDOM_SCENES": 20,
}

class SceneGenerator:
    def __init__(self):
        self.templates_path = Path(CONFIG["TMP_PATH"])
        self.objects_path = Path(CONFIG["OBJ_PATH"])
        self.json_path = Path(CONFIG["JSON_PATH"])
        self.h5_path = Path(CONFIG["H5_PATH"])
        self.temp_mesh_path = Path(CONFIG["TEMP_MESH_PATH"])
        self.gt_path = Path(CONFIG["GT_PATH"])

        # Создаем необходимые директории
        self.json_path.mkdir(parents=True, exist_ok=True)
        self.h5_path.mkdir(parents=True, exist_ok=True)
        self.temp_mesh_path.mkdir(parents=True, exist_ok=True)
        self.gt_path.mkdir(parents=True, exist_ok=True)
        
        # Собираем все доступные объекты
        self.available_objects = self._collect_objects()

    def _collect_objects(self) -> List[tuple]:
        """Собирает все доступные .obj файлы из директории objects"""
        objects = []
        for obj_file in glob.glob(str(self.objects_path / "**/*.obj"), recursive=True):
            rel_path = os.path.relpath(obj_file, str(self.objects_path))
            obj_name = Path(rel_path).stem
            objects.append((obj_name, rel_path))
        return objects

    def generate_scene(
        self,
        scene_name: str,
        object_name: str,
        object_path: str
    ) -> Dict:
        """Генерирует сцену с заданным объектом"""


        scene = {
            "scene_name": "cornell box",
            "version": "1.0",
            "objects": {
                "background_0": {
                    "mesh_path": f"{CONFIG['BASE_DIR']}/templates/backgrounds/plane.obj",
                    "transform": {
                        "translation": [
                            0.0,
                            0.0,
                            0.0
                        ],
                        "rotation": [
                            0.0,
                            0.0,
                            0.0
                        ],
                        "scale": [
                            0.5,
                            0.5,
                            0.5
                        ],
                        "normalize": False
                    },
                    "material": {
                        "diffuse": [
                            0.4,
                            0.4,
                            0.4
                        ],
                        "specular": [
                            0.01,
                            0.01,
                            0.01
                        ],
                        "random_diffuse_max": 0.4,
                        "roughness": 0.99,
                        "emissive": [
                            0.0,
                            0.0,
                            0.0
                        ],
                        "smooth_shading": True,
                        "rand_tri_diffuse_seed": None
                    }
                },
                "background_1": {
                    "mesh_path": f"{CONFIG['BASE_DIR']}/templates/backgrounds/wall0.obj",
                    "transform": {
                        "translation": [
                            0.0,
                            0.0,
                            0.0
                        ],
                        "rotation": [
                            0.0,
                            0.0,
                            0.0
                        ],
                        "scale": [
                            0.5,
                            0.5,
                            0.5
                        ],
                        "normalize": False
                    },
                    "material": {
                        "diffuse": [
                            0.8,
                            0.8,
                            0.8
                        ],
                        "specular": [
                            0.01,
                            0.01,
                            0.01
                        ],
                        "random_diffuse_max": 0.4,
                        "roughness": 0.99,
                        "emissive": [
                            0.0,
                            0.0,
                            0.0
                        ],
                        "smooth_shading": True,
                        "rand_tri_diffuse_seed": None
                    }
                },
                "background_2": {
                    "mesh_path": f"{CONFIG['BASE_DIR']}/templates/backgrounds/wall1.obj",
                    "transform": {
                        "translation": [
                            0.0,
                            0.0,
                            0.0
                        ],
                        "rotation": [
                            0.0,
                            0.0,
                            0.0
                        ],
                        "scale": [
                            0.5,
                            0.5,
                            0.5
                        ],
                        "normalize": False
                    },
                    "material": {
                        "diffuse": [
                            0.0,
                            0.5,
                            0.0
                        ],
                        "specular": [
                            0.01,
                            0.01,

                            0.01
                        ],
                        "random_diffuse_max": 0.5,
                        "roughness": 0.99,
                        "emissive": [
                            0.0,
                            0.0,
                            0.0
                        ],
                        "smooth_shading": True,
                        "rand_tri_diffuse_seed": None
                    }
                },
                "background_3": {
                    "mesh_path": f"{CONFIG['BASE_DIR']}/templates/backgrounds/wall2.obj",
                    "transform": {
                        "translation": [
                            0.0,
                            0.0,
                            0.0
                        ],
                        "rotation": [
                            0.0,
                            0.0,
                            0.0
                        ],
                        "scale": [
                            0.5,
                            0.5,
                            0.5
                        ],
                        "normalize": False
                    },
                    "material": {
                        "diffuse": [
                            0.5,
                            0.0,
                            0.0
                        ],
                        "specular": [
                            0.01,
                            0.01,
                            0.01
                        ],
                        "random_diffuse_max": 0.5,
                        "roughness": 0.99,
                        "emissive": [
                            0.0,
                            0.0,
                            0.0
                        ],
                        "smooth_shading": True,
                        "rand_tri_diffuse_seed": None
                    }
                },
                "object_0": {
                    "mesh_path": f"{CONFIG['BASE_DIR']}/objects/{object_path}",
                    "transform": {
                        "translation": [
                            random.uniform(-0.3, 0.3),
                            random.uniform(-0.3, 0.3),
                            random.uniform(-0.3, 0.3)
                        ],
                        "rotation": [
                            random.uniform(0, 360),
                            random.uniform(0, 360),
                            random.uniform(0, 360)
                        ],
                        "scale": [
                            random.uniform(0.3, 0.5),
                            random.uniform(0.3, 0.5),
                            random.uniform(0.3, 0.5)
                        ],
                        "normalize": False
                    },
                    "material": {
                        "diffuse": [
                            0.5,
                            0.5,
                            0.5
                        ],
                        "specular": [
                            0.5,
                            0.5,
                            0.5
                        ],
                        "random_diffuse_max": 0.5,
                        "roughness": random.uniform(0.001, 1.0),
                        "emissive": [
                            0.0,
                            0.0,
                            0.0
                        ],
                        "smooth_shading": True,
                        "rand_tri_diffuse_seed": None
                    }
                },
                "light_0": {
                    "mesh_path": f"{CONFIG['BASE_DIR']}/templates/lighting/tri.obj",
                    "transform": {
                        "translation": [
                            0.0,
                            0.0,
                            random.uniform(1.5, 2.5)
                        ],
                        "rotation": [
                            0.0,
                            0.0,
                            0.0

                        ],
                        "scale": [
                            2.5,
                            2.5,
                            2.5
                        ],
                        "normalize": False
                    },
                    "material": {
                        "diffuse": [
                            1.0,
                            1.0,
                            1.0
                        ],
                        "specular": [
                            0.0,
                            0.0,
                            0.0
                        ],
                        "random_diffuse_max": 0.0,
                        "roughness": 1.0,
                        "emissive": [
                            5000.0,
                            5000.0,
                            5000.0
                        ],
                        "smooth_shading": True,
                        "rand_tri_diffuse_seed": None
                    }
                }
            },
            "cameras": [
                {
                    "position": [
                        0.0,
                        -2.0,
                        0.0
                    ],
                    "look_at": [
                        0.0,
                        0.0,
                        0.0
                    ],
                    "up": [
                        0.0,
                        0.0,
                        1.0
                    ],
                    "fov": random.uniform(30, 60)
                }
            ]
        }
        
        return scene

    def save_scene(self, scene: Dict, scene_name: str):
        """Сохраняет сцену в JSON и конвертирует в H5"""
        # Сохраняем JSON
        json_path = self.json_path / f"{scene_name}.json"
        with open(json_path, 'w') as f:
            json.dump(scene, f, indent=4)
            
        # Создаем директорию для split-файлов
        split_dir = self.temp_mesh_path / "split"
        split_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Создаем конфиг сцены
            scene_config = from_dict(
                data_class=SceneConfig, 
                data=scene, 
                config=Config(check_types=True, strict=True)
            )
            
            # Генерируем временный меш
            temp_mesh_file = self.temp_mesh_path / f"{scene_name}.obj"
            
            # Генерируем меш сцены (это создаст split-файлы)
            # Используем корневую директорию проекта как base_dir
            project_root = Path(__file__).parent
            generate_scene_mesh(
                scene_config, 
                str(temp_mesh_file),
                str(project_root)
            )
            
            # Сохраняем H5
            h5_path = self.h5_path / f"{scene_name}.h5"
            save_to_h5(scene_config, str(temp_mesh_file), str(h5_path))

            # render_scene_from_json(json_path, self.gt_path)
            # Рендерим GT используя внешний скрипт
            render_script = Path(__file__).parent / "scene_processor" / "render_scene.py"
            base_dir = CONFIG["BASE_DIR"]
            cmd = f"blenderproc run {render_script} {json_path} {base_dir} {self.gt_path} {scene_name}.png"
            result = os.system(cmd)

            if result != 0:
                print(f"Warning: Rendering failed for scene {scene_name}")

            # Удаляем временные файлы
            if temp_mesh_file.exists():
                temp_mesh_file.unlink()
            
            # Удаляем split-файлы
            for split_file in split_dir.glob("*.obj"):
                split_file.unlink()
                
            print(f"Generated scene {scene_name}:")
            print(f"  - JSON: {json_path}")
            print(f"  - H5: {h5_path}")
            print(f"  - GT: {self.gt_path}")
            
        except Exception as e:
            print(f"Error converting scene {scene_name} to H5: {str(e)}")
            print(f"JSON file still saved at: {json_path}")
            # Добавим более подробный вывод ошибки для отладки
            import traceback
            print("Detailed error:")
            print(traceback.format_exc())

def main():
    generator = SceneGenerator()


    # Генерируем случайные сцены
    for i in range(CONFIG["NUM_RANDOM_SCENES"]):
        obj_name, obj_path = random.choice(generator.available_objects)
        print(obj_name, obj_path)
        scene = generator.generate_scene(
            scene_name=f"random_scene_{i}_{obj_name}",
            object_name=obj_name,
            object_path=obj_path
        )
        generator.save_scene(scene, f"random_scene_{i}_{obj_name}")

if __name__ == "__main__":
    main()
