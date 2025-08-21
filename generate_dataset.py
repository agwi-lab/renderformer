import json
import random
import os
import asyncio
from pathlib import Path
from typing import Dict, List
import glob
from scene_processor.json_h5_convertor import save_dict_to_h5

CONFIG = {
    "DATA_PATH": "/home/devel/.draft/renderformer/datasets",
    "JSON_PATH": "/home/devel/.draft/renderformer/datasets/json",
    "H5_PATH": "/home/devel/.draft/renderformer/datasets/h5",
    "GT_PATH": "/home/devel/.draft/renderformer/datasets/gt",
    "TEMP_MESH_PATH": "/home/devel/.draft/renderformer/datasets/temp",
    "OBJ_PATH": "/home/devel/.draft/renderformer/examples/objects",
    "BASE_DIR": "/home/devel/.draft/renderformer/examples",
    "SCRIPT_NAME": "render_scene.py",
    "NUM_RANDOM_SCENES": 30,
    "MAX_CONCURRENT_TASKS": 4,
}

class SceneGenerator:
    def __init__(self):
        self.objects_path = Path(CONFIG["OBJ_PATH"])
        self.json_path = Path(CONFIG["JSON_PATH"])
        self.h5_path = Path(CONFIG["H5_PATH"])
        self.gt_path = Path(CONFIG["GT_PATH"])

        # Create necessary directories
        self.json_path.mkdir(parents=True, exist_ok=True)
        self.h5_path.mkdir(parents=True, exist_ok=True)
        self.gt_path.mkdir(parents=True, exist_ok=True)
        
        # Collect all available objects
        self.available_objects = self._collect_objects()

    def _collect_objects(self) -> List[tuple]:
        """Collect all available .obj files from objects directory"""
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
        """Generate a scene with the specified object"""


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
                            random.uniform(0.4, 0.8),
                            random.uniform(0.4, 0.8),
                            random.uniform(0.4, 0.8),
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

    async def save_scene_async(self, scene: Dict, scene_name: str):
        """Asynchronously save scene to JSON and convert to H5"""
        # Save JSON
        json_path = self.json_path / f"{scene_name}.json"
        with open(json_path, 'w') as f:
            json.dump(scene, f, indent=4)

        try:
            # Save H5
            h5_path = self.h5_path / f"{scene_name}.h5"
            save_dict_to_h5(scene, h5_path)

            # Render GT using external script (asynchronously)
            render_script = Path(__file__).parent / "scene_processor" / CONFIG["SCRIPT_NAME"]
            cmd = f"blenderproc run {render_script} -j {json_path} -o {self.gt_path} -i {scene_name}.png"
            print(f"Starting render for {scene_name}: {cmd}")
            
            # Start process asynchronously
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for process completion
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                print(f"Warning: Rendering failed for scene {scene_name}")
                print(f"Stderr: {stderr.decode()}")
            else:
                print(f"Successfully rendered scene {scene_name}")

            print(f"Generated scene {scene_name}:")
            print(f"  - JSON: {json_path}")
            print(f"  - H5: {h5_path}")
            print(f"  - GT: {self.gt_path}")
            
        except Exception as e:
            print(f"Error converting scene {scene_name} to H5: {str(e)}")
            print(f"JSON file still saved at: {json_path}")
            import traceback
            print("Detailed error:")
            print(traceback.format_exc())

    async def _generate_scene_task(self, scene_index: int):
        """Task for generating a single scene"""
        obj_name, obj_path = random.choice(self.available_objects)
        print(f"Generating scene {scene_index} with {obj_name}")
        
        scene = self.generate_scene(
            scene_name=f"random_scene_{scene_index}_{obj_name}",
            object_name=obj_name,
            object_path=obj_path
        )
        
        await self.save_scene_async(scene, f"random_scene_{scene_index}_{obj_name}")
        return scene_index

    async def generate_dataset(self):
        """Asynchronously generate the entire dataset"""
        print(f"Starting async dataset generation with {CONFIG['NUM_RANDOM_SCENES']} scenes")
        print(f"Max concurrent tasks: {CONFIG['MAX_CONCURRENT_TASKS']}")
        
        # Create semaphore to limit number of concurrent tasks
        semaphore = asyncio.Semaphore(CONFIG["MAX_CONCURRENT_TASKS"])
        
        async def limited_task(scene_index):
            async with semaphore:
                return await self._generate_scene_task(scene_index)
        
        # Create tasks for all scenes
        tasks = [
            limited_task(i) for i in range(CONFIG["NUM_RANDOM_SCENES"])
        ]
        
        # Start all tasks and wait for completion
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        successful = 0
        failed = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Scene {i} failed with error: {result}")
                failed += 1
            else:
                print(f"Scene {i} completed successfully")
                successful += 1
        
        print(f"Dataset generation completed: {successful} successful, {failed} failed")

    def generate_dataset_sync(self):
        """Synchronous version for backward compatibility"""
        print("Running sync dataset generation...")
        for i in range(CONFIG["NUM_RANDOM_SCENES"]):
            obj_name, obj_path = random.choice(self.available_objects)
            print(f"Generating scene {i} with {obj_name}")
            
            scene = self.generate_scene(
                scene_name=f"random_scene_{i}_{obj_name}",
                object_name=obj_name,
                object_path=obj_path
            )
            
            # Use synchronous call for saving
            json_path = self.json_path / f"random_scene_{i}_{obj_name}.json"
            with open(json_path, 'w') as f:
                json.dump(scene, f, indent=4)
            
            h5_path = self.h5_path / f"random_scene_{i}_{obj_name}.h5"
            save_dict_to_h5(scene, h5_path)
            
            # Synchronous rendering
            render_script = Path(__file__).parent / "scene_processor" / CONFIG["SCRIPT_NAME"]
            cmd = f"blenderproc run {render_script} -j {json_path} -o {self.gt_path} -i random_scene_{i}_{obj_name}.png"
            result = os.system(cmd)
            
            if result != 0:
                print(f"Warning: Rendering failed for scene {i}")
            
            print(f"Generated scene {i}: {obj_name}")


async def main():
    generator = SceneGenerator()

    # Asynchronous generation
    await generator.generate_dataset()

    # Synchronous generation
    # generator.generate_dataset_sync()


if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())