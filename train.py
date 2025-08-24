import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import h5py
import imageio
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc

from renderformer import RenderFormerRenderingPipeline, RenderFormer
from renderformer.utils.ray_generator import RayGenerator
from renderformer.utils.transform import trans_to_cam_coord


def load_single_h5_data(file_path):
    """Загружает данные из H5 файла"""
    with h5py.File(file_path, 'r') as f:
        triangles = torch.from_numpy(np.array(f['triangles']).astype(np.float32))
        num_tris = triangles.shape[0]
        texture = torch.from_numpy(np.array(f['texture']).astype(np.float32))
        mask = torch.ones(num_tris, dtype=torch.bool)
        vn = torch.from_numpy(np.array(f['vn']).astype(np.float32))
        c2w = torch.from_numpy(np.array(f['c2w']).astype(np.float32))
        fov = torch.from_numpy(np.array(f['fov']).astype(np.float32))

        data = {
            'triangles': triangles,
            'texture': texture,
            'mask': mask,
            'c2w': c2w,
            'fov': fov,
            'vn': vn,
        }
    return data


class TrainableRenderFormerPipeline(RenderFormerRenderingPipeline):
    """Расширенный пайплайн для обучения с поддержкой градиентов"""
    
    def __init__(self, model: RenderFormer):
        super().__init__(model)
        
    def render_with_gradients(
        self,
        triangles,
        texture,
        mask,
        vn,
        c2w,
        fov,
        resolution: int = 256,  # Уменьшаем разрешение для экономии памяти
        torch_dtype: torch.dtype = torch.float16  # Используем fp16 для экономии памяти
    ):
        """
        Рендеринг с поддержкой градиентов для обучения
        """
        bs, nv = c2w.shape[0], c2w.shape[1]

        # Process data according to config
        if self.config.texture_encode_patch_size == 1 and texture.dim() == 5:
            texture = texture[:, :, :, 0, 0]

        # Log encode lighting if not learning LDR directly
        if not self.config.use_ldr:
            texture = texture.clone()
            texture[:, :, -3:] = torch.log10(texture[:, :, -3:] + 1.)

        # Handle view transformation
        if self.config.turn_to_cam_coord:
            c2w_reshaped = c2w.reshape(-1, 4, 4)
            triangles_repeated = torch.repeat_interleave(triangles, nv, dim=0)
            
            tris_for_view_tf, c2w_for_view_tf, _ = trans_to_cam_coord(
                c2w_reshaped,
                triangles_repeated
            )
            c2w_for_view_tf = c2w_for_view_tf.reshape(bs, nv, 4, 4)
            tris_for_view_tf = tris_for_view_tf.reshape(bs, nv, -1, 3, 3)
        else:
            tris_for_view_tf = triangles.unsqueeze(1).expand(-1, nv, -1, -1, -1)
            c2w_for_view_tf = c2w

        # Generate rays
        rays_o, rays_d = self.ray_generator(c2w_for_view_tf, fov / 180. * torch.pi, resolution)

        # Set precision
        assert torch_dtype in [torch.bfloat16, torch.float16, torch.float32], f"Invalid precision: {torch_dtype}"
        tf32_view_tf = torch_dtype == torch.bfloat16 or torch_dtype == torch.float16

        # Perform rendering WITH gradients (убираем torch.no_grad())
        # Используем gradient checkpointing для экономии памяти
        with torch.autocast(device_type=self.device.type, dtype=torch_dtype, enabled=(torch_dtype != torch.float32)):
            rendered_imgs = self.model(
                triangles.reshape(bs, -1, 9),
                texture,
                mask,
                vn.reshape(bs, -1, 9),
                rays_o=rays_o,
                rays_d=rays_d,
                tri_vpos_view_tf=tris_for_view_tf.reshape(bs, nv, -1, 9),
                tf32_view_tf=tf32_view_tf,
            )

        # Process output
        rendered_imgs = rendered_imgs.permute(0, 1, 3, 4, 2)

        # Log decode lighting if needed
        if not self.config.use_ldr:
            rendered_imgs = torch.pow(10., rendered_imgs) - 1.

        return rendered_imgs


class RenderFormerDataset(Dataset):
    def __init__(self, h5_dir, gt_dir, device='cuda', max_resolution=256):
        """
        Dataset для обучения RenderFormer
        
        Args:
            h5_dir: путь к директории с H5 файлами сцен
            gt_dir: путь к директории с ground truth изображениями
            device: устройство для загрузки данных
            max_resolution: максимальное разрешение для экономии памяти
        """
        self.h5_dir = Path(h5_dir)
        self.gt_dir = Path(gt_dir)
        self.device = device
        self.max_resolution = max_resolution
        
        # Собираем все H5 файлы
        self.h5_files = list(self.h5_dir.glob("*.h5"))
        print(f"Найдено {len(self.h5_files)} H5 файлов для обучения")

    def __len__(self):
        return len(self.h5_files)
    
    def __getitem__(self, idx):
        h5_file = self.h5_files[idx]
        base_name = h5_file.stem

        # Загружаем данные сцены из H5
        scene_data = load_single_h5_data(str(h5_file))
        
        # Загружаем ground truth изображения
        gt_images = []
        nv = scene_data['c2w'].shape[0]  # количество видов
        
        for view_idx in range(nv):
            gt_path = self.gt_dir / f"{base_name}.png"
            if gt_path.exists():
                gt_img = imageio.v3.imread(str(gt_path))
                gt_img = gt_img.astype(np.float32) / 255.0
                
                # Изменяем размер изображения для экономии памяти
                if gt_img.shape[0] > self.max_resolution or gt_img.shape[1] > self.max_resolution:
                    import cv2
                    gt_img = cv2.resize(gt_img, (self.max_resolution, self.max_resolution))
                
                gt_images.append(torch.from_numpy(gt_img.astype(np.float32)))
            else:
                print(f"Предупреждение: GT изображение не найдено: {gt_path}")
                # Создаем пустое изображение как заглушку
                gt_images.append(torch.zeros(self.max_resolution, self.max_resolution, 3, dtype=torch.float32))
        
        gt_images = torch.stack(gt_images)  # [num_views, H, W, 3]

        # Scene data parsing
        triangles = scene_data['triangles']
        texture = scene_data['texture']
        mask = scene_data['mask']
        vn = scene_data['vn']
        c2w = scene_data['c2w']
        fov = scene_data['fov']

        return {
            'triangles': triangles,
            'texture': texture,
            'mask': mask,
            'vn': vn,
            'c2w': c2w,
            'fov': fov,
            'gt_images': gt_images,
            'resolution': self.max_resolution
        }


class RenderFormerTrainer:
    def __init__(self, pipeline, device='cuda', learning_rate=1e-5):
        self.pipeline = pipeline
        self.device = device
        self.pipeline.to(device)
        
        # Переводим модель в режим обучения
        self.pipeline.model.train()
        
        # Включаем gradient checkpointing для экономии памяти
        if hasattr(self.pipeline.model, 'gradient_checkpointing_enable'):
            self.pipeline.model.gradient_checkpointing_enable()
        
        # Оптимизатор с меньшим learning rate
        self.optimizer = optim.AdamW(
            self.pipeline.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-4,
            eps=1e-8
        )

        # Функция потерь
        self.criterion = nn.MSELoss()
        
        # Планировщик обучения
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # TensorBoard для логирования
        self.writer = SummaryWriter('runs/renderformer_training')
        
        self.train_losses = []
        self.val_losses = []
        
        # Scaler для mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
    def clear_cache(self):
        """Очищаем кэш GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def train_epoch(self, dataloader, epoch):
        self.pipeline.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        successful_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Эпоха {epoch+1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Очищаем кэш перед каждым батчем
                self.clear_cache()
                
                # Перемещаем данные на устройство
                triangles = batch['triangles'].to(self.device, non_blocking=True)
                texture = batch['texture'].to(self.device, non_blocking=True)
                mask = batch['mask'].to(self.device, non_blocking=True)
                vn = batch['vn'].to(self.device, non_blocking=True)
                c2w = batch['c2w'].to(self.device, non_blocking=True)
                fov = batch['fov'].unsqueeze(-1).to(self.device, non_blocking=True)
                gt_images = batch['gt_images'].to(self.device, non_blocking=True)

                # Обнуляем градиенты
                self.optimizer.zero_grad()
                
                # Используем mixed precision для экономии памяти
                with torch.cuda.amp.autocast():
                    # Используем метод с поддержкой градиентов
                    rendered_images = self.pipeline.render_with_gradients(
                        triangles=triangles,
                        texture=texture,
                        mask=mask,
                        vn=vn,
                        c2w=c2w,
                        fov=fov,
                        resolution=batch['resolution'][0].item(),
                        torch_dtype=torch.float16
                    )

                    # Вычисляем потери
                    loss = self.criterion(rendered_images, gt_images)
                
                # Проверяем что loss не NaN и требует градиенты
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"ВНИМАНИЕ: loss содержит NaN или Inf: {loss.item()}")
                    continue
                    
                if not loss.requires_grad:
                    print("ВНИМАНИЕ: loss не требует градиенты!")
                    continue
                
                # Обратный проход с gradient scaling
                self.scaler.scale(loss).backward()
                
                # Проверяем градиенты
                total_grad_norm = 0
                param_count = 0
                for param in self.pipeline.model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                        param_count += 1
                
                if param_count > 0:
                    total_grad_norm = total_grad_norm ** 0.5
                else:
                    print("ВНИМАНИЕ: Нет параметров с градиентами!")
                    continue
                
                if total_grad_norm == 0:
                    print("ВНИМАНИЕ: Градиенты равны нулю!")
                    continue
                
                # Обрезаем градиенты
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.pipeline.model.parameters(), max_norm=1.0)
                
                # Обновляем веса
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                successful_batches += 1
                
                # Обновляем прогресс-бар
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Avg Loss': f'{total_loss/successful_batches:.6f}' if successful_batches > 0 else '0.000000',
                    'Grad Norm': f'{total_grad_norm:.6f}',
                    'Success': f'{successful_batches}/{batch_idx+1}'
                })
                
                # Логируем в TensorBoard
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
                self.writer.add_scalar('Grad_Norm/Train', total_grad_norm, global_step)
                
                # Освобождаем память
                del triangles, texture, mask, vn, c2w, fov, gt_images, rendered_images, loss
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA OOM в батче {batch_idx}: {e}")
                self.clear_cache()
                continue
            except Exception as e:
                print(f"Ошибка в батче {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                self.clear_cache()
                continue
        
        avg_loss = total_loss / successful_batches if successful_batches > 0 else float('inf')
        self.train_losses.append(avg_loss)
        
        # Логируем среднюю потерю за эпоху
        self.writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
        self.writer.add_scalar('Successful_Batches/Train', successful_batches, epoch)
        
        print(f"Успешно обработано батчей: {successful_batches}/{num_batches}")
        
        return avg_loss
    
    def validate(self, dataloader, epoch):
        self.pipeline.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        successful_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc='Валидация')):
                try:
                    self.clear_cache()
                    
                    # Перемещаем данные на устройство
                    triangles = batch['triangles'].to(self.device, non_blocking=True)
                    texture = batch['texture'].to(self.device, non_blocking=True)
                    mask = batch['mask'].to(self.device, non_blocking=True)
                    vn = batch['vn'].to(self.device, non_blocking=True)
                    c2w = batch['c2w'].to(self.device, non_blocking=True)
                    fov = batch['fov'].unsqueeze(-1).to(self.device, non_blocking=True)
                    gt_images = batch['gt_images'].to(self.device, non_blocking=True)
                    
                    # Для валидации используем обычный метод рендеринга
                    rendered_images = self.pipeline.render(
                        triangles=triangles,
                        texture=texture,
                        mask=mask,
                        vn=vn,
                        c2w=c2w,
                        fov=fov,
                        resolution=batch['resolution'][0].item(),
                        torch_dtype=torch.float16
                    )
                    
                    # Вычисляем потери
                    loss = self.criterion(rendered_images, gt_images)
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_loss += loss.item()
                        successful_batches += 1
                    
                    # Освобождаем память
                    del triangles, texture, mask, vn, c2w, fov, gt_images, rendered_images, loss
                    
                except torch.cuda.OutOfMemoryError as e:
                    print(f"CUDA OOM в валидационном батче {batch_idx}: {e}")
                    self.clear_cache()
                    continue
                except Exception as e:
                    print(f"Ошибка в валидационном батче {batch_idx}: {e}")
                    self.clear_cache()
                    continue
        
        avg_loss = total_loss / successful_batches if successful_batches > 0 else float('inf')
        self.val_losses.append(avg_loss)

        # Логируем валидационную потерю
        self.writer.add_scalar('Loss/Validation', avg_loss, epoch)
        self.writer.add_scalar('Successful_Batches/Validation', successful_batches, epoch)
        
        print(f"Валидация: успешно обработано батчей: {successful_batches}/{num_batches}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch, filepath):
        """Сохраняем чекпоинт модели"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.pipeline.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, filepath)
        print(f"Чекпоинт сохранен: {filepath}")

    def load_checkpoint(self, filepath):
        """Загружаем чекпоинт модели"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.pipeline.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        return checkpoint['epoch']

    def plot_losses(self):
        """Строим график потерь"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Эпоха')
        plt.ylabel('Потеря')
        plt.title('График потерь обучения')
        plt.legend()
        plt.grid(True)
        plt.show()


# Настройка обучения
def setup_training(device='cuda', max_resolution=256):
    # Пути к данным
    h5_dir = "/home/devel/.draft/renderformer/datasets/h5"
    gt_dir = "/home/devel/.draft/renderformer/datasets/gt"
    
    # Создаем датасет с ограниченным разрешением
    dataset = RenderFormerDataset(h5_dir, gt_dir, device=device, max_resolution=max_resolution)

    # Разделяем на train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Создаем DataLoader'ы с меньшим batch_size и без num_workers для экономии памяти
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Размер обучающего набора: {len(train_dataset)}")
    print(f"Размер валидационного набора: {len(val_dataset)}")
    
    return train_loader, val_loader


# Запуск обучения
def train_model(model_id="microsoft/renderformer-v1.1-swin-large", num_epochs=50, device='cuda'):
    # Настраиваем переменные окружения для экономии памяти
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Загружаем модель
    print("Загружаем модель...")
    base_model = RenderFormer.from_pretrained(model_id)
    
    # Создаем обучаемый пайплайн
    pipeline = TrainableRenderFormerPipeline(base_model)
    
    # Настраиваем данные с уменьшенным разрешением
    train_loader, val_loader = setup_training(device, max_resolution=256)

    # Создаем тренер с меньшим learning rate
    trainer = RenderFormerTrainer(pipeline, device=device, learning_rate=5e-6)

    # Создаем директорию для чекпоинтов
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    best_val_loss = float('inf')

    print("Начинаем обучение...")
    
    for epoch in range(num_epochs):
        print(f"\n=== Эпоха {epoch+1}/{num_epochs} ===")
        
        # Обучение
        train_loss = trainer.train_epoch(train_loader, epoch)
        print(f"Средняя потеря обучения: {train_loss:.6f}")
    
        # Валидация
        if len(val_loader) > 0:
            val_loss = trainer.validate(val_loader, epoch)
            print(f"Средняя потеря валидации: {val_loss:.6f}")
            
            # Сохраняем лучшую модель
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save_checkpoint(epoch, checkpoint_dir / "best_model.pth")
                print(f"Новая лучшая модель сохранена! Val Loss: {val_loss:.6f}")

        # Обновляем планировщик
        trainer.scheduler.step()
        
        # Сохраняем чекпоинт каждые 5 эпох
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(epoch, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth")

    # Строим график потерь
    trainer.plot_losses()

    # Закрываем TensorBoard writer
    trainer.writer.close()

    print("Обучение завершено!")
    return trainer


if __name__ == "__main__":
    # Настройка устройства
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Проверяем доступную память GPU
    if torch.cuda.is_available():
        print(f"Доступная память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Свободная память GPU: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # Запуск обучения
    trainer = train_model(num_epochs=3, device=device)