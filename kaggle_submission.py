!pip install -q segmentation-models-pytorch
!pip install -q bruges

import os
import torch
import numpy as np
import pandas as pd
from torch import nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from bruges.attribute import envelope, instantaneous_phase, instantaneous_frequency

def preprocess_seismic_with_attributes(shot_gather_tensor, dt=0.001):
    shot_gather_np = shot_gather_tensor.cpu().numpy()
    target_shape = shot_gather_np.shape
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        env = np.nan_to_num(envelope(shot_gather_np))
        phase = np.nan_to_num(instantaneous_phase(shot_gather_np))
        freq = np.nan_to_num(instantaneous_frequency(shot_gather_np, dt=dt))

    if freq.shape != target_shape:
        padding_needed = target_shape[0] - freq.shape[0]
        freq = np.pad(freq, ((0, padding_needed), (0, 0)), 'edge')

    def normalize(arr):
        min_val, max_val = arr.min(), arr.max()
        return (arr - min_val) / (max_val - min_val) if max_val > min_val else arr

    processed_channels = np.stack([
        normalize(shot_gather_np), normalize(env), normalize(phase), normalize(freq)
    ], axis=0)
    
    return torch.from_numpy(processed_channels).float()

class SimpleUnet(nn.Module):
    def __init__(self, in_channels=4, out_classes=1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet18", encoder_weights=None,
            in_channels=in_channels, classes=out_classes,
            activation='sigmoid'
        )
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = '/kaggle/input/best-unet-model/best_unet_model_20250608_135026.pth'
    PATH_TO_TEST_DATA = '/kaggle/input/waveform-inversion/test/'
    
    VMIN, VMAX = 1500.0, 4500.0
    VELOCITY_RANGE = VMAX - VMIN
    TARGET_SHAPE = (70, 70)
    DT = 0.001

    print(f"Usando dispositivo: {DEVICE}")
    print(f"Cargando modelo desde: {MODEL_PATH}")

    try:
        model = SimpleUnet(in_channels=4, out_classes=1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        exit()

    submission_rows = []
    
    try:
        test_files = [f for f in os.listdir(PATH_TO_TEST_DATA) if f.endswith('.npy')]
        print(f"\nIniciando inferencia en {len(test_files)} archivos de test...")

        with torch.no_grad():
            for filename in tqdm(test_files, desc="Procesando archivos de test"):
                oid = filename.replace('.npy', '')
                seis_path = os.path.join(PATH_TO_TEST_DATA, filename)
                seis_batch = np.load(seis_path)
                if seis_batch.ndim == 4:
                    shot_gather = torch.from_numpy(seis_batch[0, 0]).float()
                elif seis_batch.ndim == 3:
                    shot_gather = torch.from_numpy(seis_batch[0]).float()
                elif seis_batch.ndim == 2:
                    shot_gather = torch.from_numpy(seis_batch).float()
                else:
                    print(f"Saltando {filename} por dimensiones inesperadas: {seis_batch.ndim}")
                    continue

                input_tensor = preprocess_seismic_with_attributes(shot_gather, dt=DT)
                resized_input = F.interpolate(input_tensor.unsqueeze(0), size=TARGET_SHAPE, mode='bilinear', align_corners=False)
                prediction_norm = model(resized_input.to(DEVICE)).cpu()
                prediction_denorm = prediction_norm.squeeze().numpy() * VELOCITY_RANGE + VMIN
                
                for y_pos in range(prediction_denorm.shape[0]):
                    oid_ypos = f"{oid}_y_{y_pos}"
                    odd_columns_values = prediction_denorm[y_pos, 1::2]
                    row = [oid_ypos] + odd_columns_values.tolist()
                    submission_rows.append(row)

        num_odd_cols = len(submission_rows[0]) - 1
        header = ['oid_ypos'] + [f'x_{i}' for i in range(1, num_odd_cols * 2, 2)]
        
        print("\nCreando DataFrame de sumisión...")
        submission_df = pd.DataFrame(submission_rows, columns=header)

        submission_df.to_csv('submission.csv', index=False)
        print("\n¡Archivo 'submission.csv' creado con éxito!")
        print(f"Número de filas: {len(submission_df)}")
        print("Primeras 5 filas:")
        print(submission_df.head())

    except Exception as e:
        print(f"Ocurrió un error durante la inferencia: {e}")
 