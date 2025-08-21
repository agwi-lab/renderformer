import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union


def save_dict_to_h5(data: Dict[str, Any], h5_path: str) -> None:
    """
    Сохраняет словарь в H5 файл.
    
    Args:
        data: Словарь с данными для сохранения
        h5_path: Путь к H5 файлу
    """
    with h5py.File(h5_path, 'w') as f:
        _write_dict_to_group(f, data)


def _write_dict_to_group(group: h5py.Group, data: Dict[str, Any]) -> None:
    """
    Рекурсивно записывает словарь в группу H5.
    
    Args:
        group: H5 группа для записи
        data: Словарь с данными
    """
    for key, value in data.items():
        if isinstance(value, dict):
            # Создаем подгруппу для вложенного словаря
            subgroup = group.create_group(key)
            _write_dict_to_group(subgroup, value)
        elif isinstance(value, (list, tuple)):
            # Конвертируем список в numpy массив
            try:
                array_data = np.array(value)
                group.create_dataset(key, data=array_data)
            except (ValueError, TypeError):
                # Если не удается конвертировать в массив, сохраняем как строку
                group.attrs[key] = str(value)
        elif isinstance(value, (int, float, bool)):
            # Сохраняем числовые значения как атрибуты
            group.attrs[key] = value
        elif isinstance(value, str):
            # Сохраняем строки как атрибуты
            group.attrs[key] = value
        elif isinstance(value, np.ndarray):
            # Сохраняем numpy массивы как датасеты
            group.create_dataset(key, data=value)
        else:
            # Для остальных типов пытаемся сохранить как строку
            group.attrs[key] = str(value)


def load_dict_from_h5(h5_path: str) -> Dict[str, Any]:
    """
    Загружает словарь из H5 файла.
    
    Args:
        h5_path: Путь к H5 файлу
        
    Returns:
        Словарь с данными
    """
    with h5py.File(h5_path, 'r') as f:
        return _read_group_to_dict(f)


def _read_group_to_dict(group: h5py.Group) -> Dict[str, Any]:
    """
    Рекурсивно читает группу H5 в словарь.
    
    Args:
        group: H5 группа для чтения
        
    Returns:
        Словарь с данными
    """
    result = {}
    
    # Читаем атрибуты
    for key, value in group.attrs.items():
        result[key] = value
    
    # Читаем датасеты и подгруппы
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            # Рекурсивно читаем подгруппу
            result[key] = _read_group_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            # Читаем датасет
            result[key] = item[()]
            # Конвертируем numpy массивы в списки для совместимости
            if isinstance(result[key], np.ndarray):
                result[key] = result[key].tolist()
    
    return result


def json_to_h5(json_path: str, h5_path: str) -> None:
    """
    Конвертирует JSON файл в H5 файл.
    
    Args:
        json_path: Путь к JSON файлу
        h5_path: Путь к H5 файлу
    """
    # Загружаем JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Сохраняем в H5
    save_dict_to_h5(data, h5_path)
    print(f"Конвертирован JSON {json_path} в H5 {h5_path}")


def h5_to_json(h5_path: str, json_path: str) -> None:
    """
    Конвертирует H5 файл в JSON файл.
    
    Args:
        h5_path: Путь к H5 файлу
        json_path: Путь к JSON файлу
    """
    # Загружаем из H5
    data = load_dict_from_h5(h5_path)
    
    # Сохраняем в JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Конвертирован H5 {h5_path} в JSON {json_path}")
