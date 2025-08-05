#!/usr/bin/env python3
"""
Скрипт для создания примеров мешей для тестирования RenderFormer.
Создает базовые геометрические формы: куб, сфера, цилиндр.
"""

import os
import numpy as np
import trimesh
from pathlib import Path


def create_cube_mesh(size: float = 1.0) -> trimesh.Trimesh:
    """Создает куб"""
    vertices = np.array([
        [-size, -size, -size],  # 0
        [size, -size, -size],   # 1
        [size, size, -size],    # 2
        [-size, size, -size],   # 3
        [-size, -size, size],   # 4
        [size, -size, size],    # 5
        [size, size, size],     # 6
        [-size, size, size],    # 7
    ])
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # задняя грань
        [1, 5, 6], [1, 6, 2],  # правая грань
        [5, 4, 7], [5, 7, 6],  # передняя грань
        [4, 0, 3], [4, 3, 7],  # левая грань
        [3, 2, 6], [3, 6, 7],  # верхняя грань
        [4, 5, 1], [4, 1, 0],  # нижняя грань
    ])
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def create_sphere_mesh(radius: float = 1.0, subdivisions: int = 2) -> trimesh.Trimesh:
    """Создает сферу"""
    # Создаем икосаэдр как основу
    mesh = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)
    return mesh


def create_cylinder_mesh(radius: float = 1.0, height: float = 2.0, segments: int = 16) -> trimesh.Trimesh:
    """Создает цилиндр"""
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=segments)
    return mesh


def create_plane_mesh(size: float = 2.0) -> trimesh.Trimesh:
    """Создает плоскость"""
    vertices = np.array([
        [-size, -size, 0],
        [size, -size, 0],
        [size, size, 0],
        [-size, size, 0],
    ])
    
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def create_torus_mesh(radius: float = 1.0, tube_radius: float = 0.3, segments: int = 16) -> trimesh.Trimesh:
    """Создает тор"""
    mesh = trimesh.creation.annulus(radius=radius, r_min=radius-tube_radius, height=0.1, sections=segments)
    return mesh


def main():
    # Создаем директорию для мешей
    mesh_dir = Path("meshes")
    mesh_dir.mkdir(exist_ok=True)
    
    print("Создание примеров мешей...")
    
    # Создаем куб
    cube = create_cube_mesh(1.0)
    cube.export(mesh_dir / "cube.obj")
    print(f"Создан куб: {mesh_dir / 'cube.obj'}")
    
    # Создаем сферу
    sphere = create_sphere_mesh(1.0, subdivisions=2)
    sphere.export(mesh_dir / "sphere.obj")
    print(f"Создана сфера: {mesh_dir / 'sphere.obj'}")
    
    # Создаем цилиндр
    cylinder = create_cylinder_mesh(0.5, 2.0, segments=16)
    cylinder.export(mesh_dir / "cylinder.obj")
    print(f"Создан цилиндр: {mesh_dir / 'cylinder.obj'}")
    
    # Создаем плоскость
    plane = create_plane_mesh(2.0)
    plane.export(mesh_dir / "plane.obj")
    print(f"Создана плоскость: {mesh_dir / 'plane.obj'}")
    
    # Создаем тор
    torus = create_torus_mesh(1.0, 0.3, segments=16)
    torus.export(mesh_dir / "torus.obj")
    print(f"Создан тор: {mesh_dir / 'torus.obj'}")
    
    print(f"\nВсе меши сохранены в директории: {mesh_dir.absolute()}")
    print("Теперь можно использовать эти меши для генерации H5 датасетов!")


if __name__ == "__main__":
    main() 