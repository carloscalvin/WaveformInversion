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

class EfficientNetB7_Unet(nn.Module):
    def __init__(self, in_channels=4, out_classes=1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="timm-efficientnet-b7",
            encoder_weights=None,
            in_channels=in_channels,
            classes=out_classes,
            activation='sigmoid'
        )
        self.final_adapter = nn.AdaptiveAvgPool2d((70, 70))
        self._adapt_encoder_to_full_resolution()

    def _adapt_encoder_to_full_resolution(self):
        try:
            encoder = self.model.encoder
            new_stride = (5, 1)
            print(f"Adaptando el 'stem' del encoder. Stride original: {encoder.conv_stem.stride}")
            encoder.conv_stem.stride = new_stride
            print(f"Nuevo stride del 'stem': {encoder.conv_stem.stride}")
        except AttributeError:
            print(f"AVISO: No se pudo modificar el 'stem' del encoder automáticamente para el backbone {self.model.encoder.name}.")

    def forward(self, x):
        raw_output = self.model(x)
        final_output = self.final_adapter(raw_output)
        return final_output

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_PATH = '/kaggle/input/best-b7-model/best_efficientnetb7_model_20250620_195611.pth'
    PATH_TO_TEST_DATA = '/kaggle/input/waveform-inversion/test/'
    
    VMIN, VMAX = 1500.0, 4500.0
    VELOCITY_RANGE = VMAX - VMIN
    DT = 0.001
    NUM_SOURCES_TO_ENSEMBLE = 5

    print(f"Usando dispositivo: {DEVICE}")
    print(f"Cargando modelo EfficientNet-B7 desde: {MODEL_PATH}")

    try:
        model = EfficientNetB7_Unet(in_channels=4, out_classes=1)
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
        print(f"Se promediarán las predicciones de {NUM_SOURCES_TO_ENSEMBLE} fuentes para cada archivo.")

        with torch.no_grad():
            for filename in tqdm(test_files, desc="Procesando archivos de test"):
                oid = filename.replace('.npy', '')
                seis_path = os.path.join(PATH_TO_TEST_DATA, filename)
                sample_seismic_data = np.load(seis_path)

                if sample_seismic_data.ndim != 3 or sample_seismic_data.shape[0] < NUM_SOURCES_TO_ENSEMBLE:
                    print(f"\nAviso: Saltando archivo {filename} por tener una forma inesperada: {sample_seismic_data.shape}")
                    continue

                source_predictions = []
                for source_idx in range(NUM_SOURCES_TO_ENSEMBLE):
                    shot_gather = torch.from_numpy(sample_seismic_data[source_idx]).float()
                    input_tensor = preprocess_seismic_with_attributes(shot_gather, dt=DT)
                    prediction_norm = model(input_tensor.unsqueeze(0).to(DEVICE)).cpu()
                    source_predictions.append(prediction_norm)

                ensembled_prediction_norm = torch.stack(source_predictions).mean(dim=0)
                
                prediction_denorm = ensembled_prediction_norm.squeeze().numpy() * VELOCITY_RANGE + VMIN
                
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
 