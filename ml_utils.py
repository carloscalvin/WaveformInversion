import torch
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

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

class SeismicDataset(Dataset):
    def __init__(self, data_family_paths, preprocess_function, target_shape=(70, 70), dt=0.001, vmin=1500.0, vmax=4500.0):
        """
        Dataset para cargar los datos sísmicos de OpenFWI.
        
        Args:
            data_family_paths (list): Lista de rutas a las carpetas de las familias (ej. ['.../FlatVel_A/']).
            preprocess_function (callable): La función que toma un sismograma y devuelve los canales de atributos.
            dt (float): Intervalo de tiempo para el pre-procesado.
        """
        super().__init__()
        self.preprocess = preprocess_function
        self.dt = dt
        self.target_shape = target_shape
        self.samples = []
        self.cache = {}
        self.vmin = vmin
        self.vmax = vmax        

        print("Buscando archivos y creando el índice del dataset...")
        for family_path in data_family_paths:
            # Detectar la estructura de la carpeta (Vel/Style vs Fault)
            is_vel_style = os.path.isdir(os.path.join(family_path, 'model'))
            
            if is_vel_style:
                model_dir = os.path.join(family_path, 'model')
                data_dir = os.path.join(family_path, 'data')
                model_files = sorted(os.listdir(model_dir))
                data_files = sorted(os.listdir(data_dir))
                
                for mf, df in zip(model_files, data_files):
                    # Para cada par de archivos, añadimos 500 muestras a nuestra lista
                    num_samples_in_file = 500 # Asumimos 500 por archivo
                    for i in range(num_samples_in_file):
                        self.samples.append({
                            'vel_path': os.path.join(model_dir, mf),
                            'seis_path': os.path.join(data_dir, df),
                            'index_in_file': i
                        })
            else: # Estructura tipo Fault
                vel_files = sorted([f for f in os.listdir(family_path) if f.startswith('vel')])
                for vf in vel_files:
                    sf = vf.replace('vel', 'seis')
                    num_samples_in_file = 500
                    for i in range(num_samples_in_file):
                         self.samples.append({
                            'vel_path': os.path.join(family_path, vf),
                            'seis_path': os.path.join(family_path, sf),
                            'index_in_file': i
                        })

        print(f"Dataset creado. Número total de muestras encontradas: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        vel_path = sample_info['vel_path']
        seis_path = sample_info['seis_path']

        if vel_path not in self.cache:
            self.cache[vel_path] = np.load(vel_path)
        vel_batch = self.cache[vel_path]

        if seis_path not in self.cache:
            self.cache[seis_path] = np.load(seis_path)
        seis_batch = self.cache[seis_path]

        # Extraer la muestra específica
        index = sample_info['index_in_file']
        velocity_map = torch.from_numpy(vel_batch[index]).float()
        shot_gather = torch.from_numpy(seis_batch[index, 0]).float() # Usamos solo la primera fuente por simplicidad

        # Aplicar el pre-procesado para obtener los 4 canales
        input_tensor = self.preprocess(shot_gather, dt=self.dt)
        
        # Añadimos una dimensión de lote para la función de interpolación
        input_tensor = input_tensor.unsqueeze(0)  # Shape ahora: (1, 4, 1000, 70)
        
        # Redimensionamos la entrada a la forma del target (70, 70)
        resized_input = F.interpolate(input_tensor, size=self.target_shape, mode='bilinear', align_corners=False)
        
        # Quitamos la dimensión de lote que añadimos
        resized_input = resized_input.squeeze(0) # Shape final: (4, 70, 70)
        target_norm = (velocity_map - self.vmin) / (self.vmax - self.vmin)

        return resized_input, target_norm

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
