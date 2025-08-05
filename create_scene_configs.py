#!/usr/bin/env python3
"""
Скрипт для создания примеров конфигураций сцен в JSON формате.
Создает различные сцены с разными материалами и камерами.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List


def create_material_config(material_type: str = "default") -> Dict:
    """Создает конфигурацию материала"""
    
    materials = {
        "default": {
            "diffuse": [0.8, 0.6, 0.4],
            "specular": [0.2, 0.2, 0.2],
            "roughness": 0.3,
            "emissive": [0.0, 0.0, 0.0],
            "smooth_shading": True
        },
        "metal": {
            "diffuse": [0.1, 0.1, 0.1],
            "specular": [0.9, 0.9, 0.9],
            "roughness": 0.1,
            "emissive": [0.0, 0.0, 0.0],
            "smooth_shading": True
        },
        "plastic": {
            "diffuse": [0.2, 0.8, 0.2],
            "specular": [0.1, 0.1, 0.1],
            "roughness": 0.8,
            "emissive": [0.0, 0.0, 0.0],
            "smooth_shading": True
        },
        "glass": {
            "diffuse": [0.9, 0.9, 0.9],
            "specular": [0.9, 0.9, 0.9],
            "roughness": 0.0,
            "emissive": [0.0, 0.0, 0.0],
            "smooth_shading": True
        },
        "emissive": {
            "diffuse": [0.1, 0.1, 0.1],
            "specular": [0.0, 0.0, 0.0],
            "roughness": 1.0,
            "emissive": [1.0, 0.8, 0.6],
            "smooth_shading": True
        }
    }
    
    return materials.get(material_type, materials["default"])


def create_transform_config(position: List[float] = [0, 0, 0], 
                          rotation: List[float] = [0, 0, 0],
                          scale: List[float] = [1, 1, 1]) -> Dict:
    """Создает конфигурацию трансформации"""
    return {
        "translation": position,
        "rotation": rotation,
        "scale": scale,
        "normalize": True
    }


def create_camera_config(position: List[float], 
                       look_at: List[float] = [0, 0, 0],
                       up: List[float] = [0, 0, 1],
                       fov: float = 60.0) -> Dict:
    """Создает конфигурацию камеры"""
    return {
        "position": position,
        "look_at": look_at,
        "up": up,
        "fov": fov
    }


def create_simple_scene_config(scene_name: str, mesh_name: str, 
                             material_type: str = "default") -> Dict:
    """Создает простую конфигурацию сцены с одним объектом"""
    
    # Создаем объект
    object_config = {
        "mesh_path": f"{mesh_name}.obj",
        "material": create_material_config(material_type),
        "transform": create_transform_config(),
        "remesh": False,
        "remesh_target_face_num": 2048
    }
    
    # Создаем камеры вокруг объекта
    cameras = []
    for i in range(8):
        angle = i * 45  # 8 камер по кругу
        radius = 3.0
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        z = 1.5
        
        camera = create_camera_config(
            position=[x, y, z],
            look_at=[0, 0, 0],
            up=[0, 0, 1],
            fov=60.0
        )
        cameras.append(camera)
    
    return {
        "scene_name": scene_name,
        "version": "1.0",
        "objects": {
            "main_object": object_config
        },
        "cameras": cameras
    }


def create_complex_scene_config(scene_name: str) -> Dict:
    """Создает сложную конфигурацию сцены с несколькими объектами"""
    
    objects = {}
    
    # Основной объект (куб)
    objects["cube"] = {
        "mesh_path": "cube.obj",
        "material": create_material_config("default"),
        "transform": create_transform_config([0, 0, 0]),
        "remesh": False,
        "remesh_target_face_num": 2048
    }
    
    # Сфера
    objects["sphere"] = {
        "mesh_path": "sphere.obj",
        "material": create_material_config("metal"),
        "transform": create_transform_config([2, 0, 0]),
        "remesh": False,
        "remesh_target_face_num": 2048
    }
    
    # Цилиндр
    objects["cylinder"] = {
        "mesh_path": "cylinder.obj",
        "material": create_material_config("plastic"),
        "transform": create_transform_config([-2, 0, 0]),
        "remesh": False,
        "remesh_target_face_num": 2048
    }
    
    # Плоскость (пол)
    objects["floor"] = {
        "mesh_path": "plane.obj",
        "material": create_material_config("default"),
        "transform": create_transform_config([0, 0, -1], [0, 0, 0], [3, 3, 1]),
        "remesh": False,
        "remesh_target_face_num": 2048
    }
    
    # Источник света
    objects["light"] = {
        "mesh_path": "sphere.obj",
        "material": create_material_config("emissive"),
        "transform": create_transform_config([0, 0, 2], [0, 0, 0], [0.1, 0.1, 0.1]),
        "remesh": False,
        "remesh_target_face_num": 2048
    }
    
    # Создаем камеры
    cameras = []
    for i in range(12):
        angle = i * 30  # 12 камер по кругу
        radius = 4.0
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        z = 2.0
        
        camera = create_camera_config(
            position=[x, y, z],
            look_at=[0, 0, 0],
            up=[0, 0, 1],
            fov=60.0
        )
        cameras.append(camera)
    
    return {
        "scene_name": scene_name,
        "version": "1.0",
        "objects": objects,
        "cameras": cameras
    }


def create_random_scene_config(scene_name: str) -> Dict:
    """Создает случайную конфигурацию сцены"""
    
    mesh_names = ["cube", "sphere", "cylinder", "torus"]
    material_types = ["default", "metal", "plastic", "glass"]
    
    objects = {}
    
    # Создаем случайное количество объектов (1-4)
    num_objects = random.randint(1, 4)
    
    for i in range(num_objects):
        mesh_name = random.choice(mesh_names)
        material_type = random.choice(material_types)
        
        # Случайная позиция
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
        z = random.uniform(-1, 1)
        
        # Случайный поворот
        rx = random.uniform(0, 360)
        ry = random.uniform(0, 360)
        rz = random.uniform(0, 360)
        
        # Случайный масштаб
        scale = random.uniform(0.5, 1.5)
        
        objects[f"object_{i}"] = {
            "mesh_path": f"{mesh_name}.obj",
            "material": create_material_config(material_type),
            "transform": create_transform_config(
                [x, y, z],
                [rx, ry, rz],
                [scale, scale, scale]
            ),
            "remesh": False,
            "remesh_target_face_num": 2048
        }
    
    # Создаем случайные камеры
    cameras = []
    num_cameras = random.randint(6, 12)
    
    for i in range(num_cameras):
        angle = random.uniform(0, 360)
        radius = random.uniform(3, 6)
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        z = random.uniform(1, 3)
        
        camera = create_camera_config(
            position=[x, y, z],
            look_at=[0, 0, 0],
            up=[0, 0, 1],
            fov=random.uniform(45, 75)
        )
        cameras.append(camera)
    
    return {
        "scene_name": scene_name,
        "version": "1.0",
        "objects": objects,
        "cameras": cameras
    }


def main():
    # Создаем директорию для конфигураций
    config_dir = Path("scene_configs")
    config_dir.mkdir(exist_ok=True)
    
    print("Создание примеров конфигураций сцен...")
    
    # Создаем простые сцены
    simple_scenes = []
    
    # Сцена с кубом
    simple_scenes.append(create_simple_scene_config("cube_scene", "cube", "default"))
    
    # Сцена со сферой
    simple_scenes.append(create_simple_scene_config("sphere_scene", "sphere", "metal"))
    
    # Сцена с цилиндром
    simple_scenes.append(create_simple_scene_config("cylinder_scene", "cylinder", "plastic"))
    
    # Сложная сцена
    complex_scene = create_complex_scene_config("complex_scene")
    
    # Случайные сцены
    random_scenes = []
    for i in range(5):
        random_scenes.append(create_random_scene_config(f"random_scene_{i}"))
    
    # Сохраняем все конфигурации
    all_scenes = simple_scenes + [complex_scene] + random_scenes
    
    config_file = config_dir / "sample_scenes.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(all_scenes, f, indent=2, ensure_ascii=False)
    
    print(f"Создано {len(all_scenes)} конфигураций сцен")
    print(f"Конфигурации сохранены в: {config_file}")
    
    # Создаем отдельные файлы для каждого типа датасета
    train_scenes = all_scenes[:6]  # Первые 6 сцен для тренировки
    test_scenes = all_scenes[6:8]  # Следующие 2 для теста
    validation_scenes = all_scenes[8:]  # Остальные для валидации
    
    # Сохраняем отдельные файлы
    with open(config_dir / "train_scenes.json", 'w', encoding='utf-8') as f:
        json.dump(train_scenes, f, indent=2, ensure_ascii=False)
    
    with open(config_dir / "test_scenes.json", 'w', encoding='utf-8') as f:
        json.dump(test_scenes, f, indent=2, ensure_ascii=False)
    
    with open(config_dir / "validation_scenes.json", 'w', encoding='utf-8') as f:
        json.dump(validation_scenes, f, indent=2, ensure_ascii=False)
    
    print(f"Созданы отдельные файлы для датасетов:")
    print(f"  - Тренировка: {len(train_scenes)} сцен")
    print(f"  - Тест: {len(test_scenes)} сцен")
    print(f"  - Валидация: {len(validation_scenes)} сцен")
    
    print(f"\nВсе файлы сохранены в директории: {config_dir.absolute()}")


if __name__ == "__main__":
    main() 