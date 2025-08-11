import blenderproc as bproc
import json
import argparse
import os
import numpy as np


def save_image_array(image_array: np.ndarray, output_path: str):
    """Сохранить изображение без matplotlib для лучшего качества"""
    # Убеждаемся что изображение в правильном формате
    if image_array.dtype != np.uint8:
        # Нормализуем значения от 0 до 255
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    
    # Используем PIL для лучшего качества сохранения
    from PIL import Image
    
    # Конвертируем из RGB в формат PIL
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        img = Image.fromarray(image_array, 'RGB')
    elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
        img = Image.fromarray(image_array, 'RGBA')
    else:
        img = Image.fromarray(image_array)
    
    img.save(output_path, quality=95, optimize=True)
    print(f"Image successfully saved to {output_path}")

def render_scene_from_json(json_path: str, base_path: str, output_path: str):
    """Render a scene from JSON description to PNG"""
    # Load JSON configuration
    with open(json_path, 'r') as f:
        scene_config = json.load(f)

    # Initialize BlenderProc
    bproc.init()
    
    # Очистить сцену от объектов по умолчанию
    bproc.utility.reset_keyframes()
    
    # Удалить дефолтный куб и свет
    for obj in bproc.object.get_all_mesh_objects():
        obj.delete()
    
    # Удалить дефолтные источники света (используем bpy напрямую)
    import bpy
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Set output directory
    os.makedirs(output_path, exist_ok=True)

    # Load all objects
    for obj_name, obj_config in scene_config["objects"].items():
        try:
            # Construct full path to mesh
            mesh_path = os.path.join(os.path.dirname(base_path), obj_config["mesh_path"])

            # Load object
            objs = bproc.loader.load_obj(mesh_path)
            if not objs:
                print(f"Warning: No objects loaded from {mesh_path}")
                continue
                
            obj = objs[0]
            
            # Apply transform
            transform = obj_config["transform"]
            if "translation" in transform:
                obj.set_location(transform["translation"])
            if "rotation" in transform:
                rotation_rad = [np.radians(angle) for angle in transform["rotation"]]
                obj.set_rotation_euler(rotation_rad)
            if "scale" in transform:
                obj.set_scale(transform["scale"])
            
            # Создать и применить материал
            mat_config = obj_config["material"]
            
            # Создать новый материал
            mat = bproc.material.create(f"material_{obj_name}")
            
            # Установить базовые свойства материала
            diffuse = mat_config.get("diffuse", [0.8, 0.8, 0.8])
            specular = mat_config.get("specular", [0.5, 0.5, 0.5])
            roughness = mat_config.get("roughness", 0.5)
            
            # Добавить альфа канал к диффузному цвету если его нет
            if len(diffuse) == 3:
                diffuse = diffuse + [1.0]
            
            mat.set_principled_shader_value("Base Color", diffuse)
            # В Blender 4.0+ используется IOR вместо Specular
            try:
                mat.set_principled_shader_value("Specular IOR Level", np.mean(specular))
            except:
                try:
                    mat.set_principled_shader_value("IOR", 1.0 + np.mean(specular))
                except:
                    print("Предупреждение: Не удалось установить значение Specular/IOR")
            mat.set_principled_shader_value("Roughness", roughness)

            # Обработка эмиссивных материалов (для источников света)
            emissive = mat_config.get("emissive", [0, 0, 0])
            if "light" in obj_name.lower() or np.any(np.array(emissive) > 0):
                # Добавить альфа канал к эмиссивному цвету
                if len(emissive) == 3:
                    emissive = emissive + [1.0]

                # В Blender 4.0+ эмиссионные свойства могут называться по-другому
                try:
                    mat.set_principled_shader_value("Emission Color", emissive[:3])
                    # Увеличить силу эмиссии для лучшего освещения
                    emission_strength = max(np.mean(emissive[:3]) * 10, 1.0) if np.mean(emissive[:3]) > 0 else 0
                    mat.set_principled_shader_value("Emission Strength", emission_strength)
                except:
                    try:
                        mat.set_principled_shader_value("Emission", emissive[:3])
                        emission_strength = max(np.mean(emissive[:3]) * 10, 1.0) if np.mean(emissive[:3]) > 0 else 0
                        mat.set_principled_shader_value("Emission Strength", emission_strength)
                    except:
                        print(f"Предупреждение: Не удалось установить эмиссионные свойства для {obj_name}")
                
                # Сделать объект источником света
                obj.enable_rigidbody(False)
            
            # Применить материал к объекту
            obj.replace_materials(mat)
                    
        except Exception as e:
            print(f"Error loading object {obj_name}: {e}")

    # Настройка камеры
    if "cameras" in scene_config and len(scene_config["cameras"]) > 0:
        cam_config = scene_config["cameras"][0]
        
        # Установить разрешение
        bproc.camera.set_resolution(512, 512)
        
        # Установить FOV
        fov_degrees = cam_config.get("fov", 37.5)
        bproc.camera.set_intrinsics_from_blender_params(fov_degrees, lens_unit="FOV")
        
        # Вычислить корректную матрицу камеры
        position = np.array(cam_config["position"])
        look_at = np.array(cam_config["look_at"])
        up = np.array(cam_config.get("up", [0, 0, 1]))
        
        # Нормализовать векторы
        forward = look_at - position
        forward = forward / np.linalg.norm(forward)
        
        # Вычислить правильную систему координат камеры
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up_corrected = np.cross(right, forward)
        up_corrected = up_corrected / np.linalg.norm(up_corrected)
        
        # Создать матрицу поворота (важно: -forward для правильной ориентации)
        rotation_matrix = np.column_stack((right, up_corrected, -forward))
        
        # Создать матрицу трансформации
        camera_pose = bproc.math.build_transformation_mat(position, rotation_matrix)
        bproc.camera.add_camera_pose(camera_pose)
    else:
        # Установить камеру по умолчанию
        print("Камера не найдена в конфигурации, использую настройки по умолчанию")
        bproc.camera.set_resolution(512, 512)
        bproc.camera.set_intrinsics_from_blender_params(37.5, lens_unit="FOV")
        
        # Позиция камеры по умолчанию
        camera_pose = bproc.math.build_transformation_mat([0, -3, 1], 
                                                         bproc.camera.rotation_from_forward_vec([0, 1, -0.3]))
        bproc.camera.add_camera_pose(camera_pose)

    # Добавить освещение окружения если нет эмиссивных объектов
    has_lights = any("light" in name.lower() or 
                    np.any(np.array(obj_config["material"].get("emissive", [0, 0, 0])) > 0)
                    for name, obj_config in scene_config["objects"].items())
    
    if not has_lights:
        print("Добавляю освещение окружения")
        # Добавить освещение окружения
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        
        # Настроить цвет фона
        world.use_nodes = False
        world.color = (0.1, 0.1, 0.15)
        
        # Добавить ключевой источник света
        light_data = bpy.data.lights.new(name="Key_Light", type='SUN')
        light_data.energy = 5.0
        light_data.color = (1.0, 1.0, 1.0)
        
        light_object = bpy.data.objects.new(name="Key_Light", object_data=light_data)
        light_object.location = (2, 2, 5)
        bpy.context.collection.objects.link(light_object)
    
    # Настройка рендерера
    try:
        bproc.renderer.set_denoiser("OPTIX")  # Попробовать OPTIX
    except:
        try:
            bproc.renderer.set_denoiser("OPENIMAGEDENOISE")  # Fallback к OIDN
        except:
            print("Предупреждение: Деноизер не доступен")
    
    bproc.renderer.set_output_format("PNG")
    bproc.renderer.set_max_amount_of_samples(128)  # Увеличить количество сэмплов для лучшего качества
    
    # Настроить Cycles параметры
    bpy.context.scene.render.engine = 'CYCLES'
    cycles = bpy.context.scene.cycles
    cycles.use_denoising = True
    cycles.max_bounces = 8
    cycles.caustics_reflective = True
    cycles.caustics_refractive = True
    
    # Настроить цветовое пространство
    bpy.context.scene.view_settings.view_transform = 'Filmic'
    bpy.context.scene.view_settings.look = 'Medium High Contrast'

    # Render scene
    data = bproc.renderer.render()

    # Save PNG
    output_png = os.path.join(output_path, "render.png")
    img_data = data['colors'][0]
    save_image_array(img_data, output_png)
    print(f"Rendered image saved to: {output_png}")

def main():
    parser = argparse.ArgumentParser(description="Render a scene from JSON using BlenderProc")
    parser.add_argument("json_path", help="Path to the scene JSON file")
    parser.add_argument("base_path", help="Path to the scene JSON file")
    parser.add_argument("output_dir", help="Directory to save rendered PNG")
    args = parser.parse_args()

    render_scene_from_json(args.json_path, args.base_path, args.output_dir)

if __name__ == "__main__":
    main()