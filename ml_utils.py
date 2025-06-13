import torch
import numpy as np
import random
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm.auto import tqdm

def calculate_mae(y_pred, y_true):
    """
    Calcula el Error Absoluto Medio (MAE) entre dos tensores de PyTorch.
    
    Args:
        y_pred (torch.Tensor): Tensor con las predicciones del modelo.
        y_true (torch.Tensor): Tensor con los valores reales (etiquetas).

    Returns:
        torch.Tensor: Un tensor escalar con el valor del MAE.
    """
    return torch.mean(torch.abs(y_pred - y_true))

class AugmentationWrapper(Dataset):
    def __init__(self, dataset, hflip_prob=0.5):
        """
        Envoltorio que aplica aumentaciones a un dataset existente.
        
        Args:
            dataset: Una instancia de un objeto Dataset de PyTorch.
            hflip_prob (float): Probabilidad de aplicar un volteo horizontal.
        """
        self.dataset = dataset
        self.hflip_prob = hflip_prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Obtener la muestra original del dataset envuelto
        input_tensor, target_norm, _ = self.dataset[idx]
        
        # Aplicar la aumentación con una cierta probabilidad
        if random.random() < self.hflip_prob:
            input_tensor = torch.flip(input_tensor, dims=[2])
            target_norm = torch.flip(target_norm, dims=[2])

        return input_tensor, target_norm

class SeismicDataset(Dataset):
    """
    Dataset optimizado que precarga todos los datos sísmicos en la RAM para evitar
    la lectura de disco en cada __getitem__ y permitir que múltiples workers
    compartan eficientemente los datos en memoria.
    """
    def __init__(self, preprocessed_data_path, preprocess_function, num_sources_per_sample=5, target_shape=(70, 70), dt=0.001, vmin=1500.0, vmax=4500.0):
        """
        Args:
            preprocessed_data_path (str): Ruta al directorio con los ficheros .npz.
            preprocess_function (callable): Función para preprocesar el sismograma.
            num_sources_per_sample (int): Número de fuentes por fichero .npz.
            ...
        """
        super().__init__()
        self.preprocess = preprocess_function
        self.dt = dt
        self.target_shape = target_shape
        self.vmin = vmin
        self.vmax = vmax
        
        # 1. Escanear todos los ficheros .npz base
        all_files = []
        print("Buscando ficheros de muestras preprocesadas...")
        for root, _, files in os.walk(preprocessed_data_path):
            for file in files:
                if file.endswith('.npz'):
                    all_files.append(os.path.join(root, file))

        # 2. Precargar todos los datos en un diccionario en RAM
        # La clave será la ruta del fichero y el valor serán los datos cargados.
        self.data_cache = {}
        print(f"Precargando {len(all_files)} ficheros en RAM. Esto puede tardar un momento...")
        for filepath in tqdm(all_files, desc="Cargando datos"):
            with np.load(filepath) as data:
                # Convertimos a tensores de PyTorch aquí mismo
                self.data_cache[filepath] = {
                    'velocity_map': torch.from_numpy(data['velocity_map']).float(),
                    'seismic_data': torch.from_numpy(data['seismic_data']).float()
                }

        # 3. Crear el índice de muestras (filepath, source_idx) como antes
        self.samples = []
        for filepath in all_files:
            for source_idx in range(num_sources_per_sample):
                self.samples.append((filepath, source_idx))
        
        print(f"\n¡Dataset precargado en RAM! {len(self.data_cache)} ficheros base cargados.")
        print(f"Número total de muestras de entrenamiento (ficheros x fuentes): {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Obtener la ruta del fichero y el índice de la fuente
        filepath, source_idx = self.samples[idx]

        # Obtener los datos del diccionario en RAM
        cached_data = self.data_cache[filepath]
        velocity_map = cached_data['velocity_map']
        seismic_data_all_sources = cached_data['seismic_data']

        # El resto del preprocesamiento sigue igual
        shot_gather = seismic_data_all_sources[source_idx]
        input_tensor = self.preprocess(shot_gather, dt=self.dt)
        resized_input = F.interpolate(input_tensor.unsqueeze(0), size=self.target_shape, mode='bilinear', align_corners=False).squeeze(0)
        target_norm = (velocity_map - self.vmin) / (self.vmax - self.vmin)
        target_norm = target_norm.unsqueeze(0)

        sample_id = f"{os.path.basename(filepath).replace('.npz', '')}_source_{source_idx}"

        return resized_input, target_norm, sample_id

def plot_training_history(train_history, val_history):
    """
    Visualiza el historial de MAE de entrenamiento y validación.
    
    Args:
        train_history (list): Lista con los MAE de entrenamiento por época.
        val_history (list): Lista con los MAE de validación por época.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label='MAE de Entrenamiento', color='blue')
    plt.plot(val_history, label='MAE de Validación', color='orange')
    plt.title('Historial de Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('MAE (m/s)')
    plt.legend()
    plt.grid(True)
    plt.show()
