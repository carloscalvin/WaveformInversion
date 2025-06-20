import os
import torch
import numpy as np
from tqdm.auto import tqdm
import model as model_loader
import ps_utils

def calculate_mae_denorm(prediction_denorm, ground_truth_denorm):
    """Calcula el MAE en la escala original (m/s)."""
    return torch.mean(torch.abs(prediction_denorm - ground_truth_denorm)).item()

def validate(model, validation_data_path, device, config):
    """
    Ejecuta la validación de un modelo sobre un conjunto de datos y calcula el MAE.

    Args:
        model (torch.nn.Module): El modelo a evaluar.
        validation_data_path (str): Ruta al directorio base de los datos de validación.
        device (str): Dispositivo para la inferencia ('cpu' o 'cuda').
        config (dict): Diccionario con parámetros de configuración.

    Returns:
        float: El MAE promedio sobre todo el conjunto de validación.
    """
    model.to(device)
    model.eval()

    all_mae_scores = []
    
    families_to_validate = [d for d in os.listdir(validation_data_path) if os.path.isdir(os.path.join(validation_data_path, d))]
    print(f"Familias encontradas para validar: {families_to_validate}")

    with torch.no_grad():
        for family in tqdm(families_to_validate, desc="Validando Familias"):
            family_path = os.path.join(validation_data_path, family)
            
            is_vel_style = os.path.isdir(os.path.join(family_path, 'model'))
            if is_vel_style:
                data_dir = os.path.join(family_path, 'data')
                model_dir = os.path.join(family_path, 'model')
                seis_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
            else:
                all_files = os.listdir(family_path)
                vel_files = [f for f in all_files if f.startswith('vel')]
                seis_files = [os.path.join(family_path, v.replace('vel', 'seis')) for v in vel_files]

            for seis_path in tqdm(seis_files, desc=f"Ficheros en {family}", leave=False):
                try:
                    if is_vel_style:
                        vel_path = seis_path.replace('/data/', '/model/').replace('data', 'model')
                    else:
                        vel_path = seis_path.replace('seis', 'vel')
                    
                    seis_file_data = np.load(seis_path)
                    vel_file_data = np.load(vel_path)

                    for sample_idx in range(seis_file_data.shape[0]):
                        sample_seismic_data = seis_file_data[sample_idx]
                        ground_truth_map = vel_file_data[sample_idx]

                        if sample_seismic_data.ndim != 3 or sample_seismic_data.shape[0] < config['NUM_SOURCES_TO_ENSEMBLE']:
                            continue

                        source_predictions = []
                        for source_idx in range(config['NUM_SOURCES_TO_ENSEMBLE']):
                            shot_gather = torch.from_numpy(sample_seismic_data[source_idx]).float()
                            input_tensor = ps_utils.preprocess_seismic_with_attributes(shot_gather, dt=config['DT'])
                            prediction_norm = model(input_tensor.unsqueeze(0).to(device)).cpu()
                            source_predictions.append(prediction_norm)

                        ensembled_prediction_norm = torch.stack(source_predictions).mean(dim=0)
                        
                        prediction_denorm = ensembled_prediction_norm.squeeze() * config['VELOCITY_RANGE'] + config['VMIN']
                        ground_truth_denorm = torch.from_numpy(ground_truth_map).squeeze()

                        mae = calculate_mae_denorm(prediction_denorm, ground_truth_denorm)
                        all_mae_scores.append(mae)

                except Exception as e:
                    print(f"Error procesando {seis_path} en el índice {sample_idx}: {e}")
                    continue

    if not all_mae_scores:
        print("No se pudo calcular ningún MAE. Revisa las rutas y el formato de los datos.")
        return float('inf')
        
    return np.mean(all_mae_scores)

if __name__ == '__main__':
    MODEL_TO_VALIDATE_PATH = 'models/efficientnet-b7_soup_model.pth'
    VALIDATION_DATA_PATH = 'kaggle/input/train_samples'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        'VMIN': 1500.0,
        'VMAX': 4500.0,
        'DT': 0.001,
        'NUM_SOURCES_TO_ENSEMBLE': 5,
    }
    config['VELOCITY_RANGE'] = config['VMAX'] - config['VMIN']

    print(f"--- Iniciando Validación ---")
    print(f"Modelo a validar: {MODEL_TO_VALIDATE_PATH}")
    print(f"Datos de validación: {VALIDATION_DATA_PATH}")
    print(f"Usando dispositivo: {DEVICE}")

    try:
        model_to_test = model_loader.SimpleUnet(
            encoder_name="timm-efficientnet-b7",
            in_channels=4,
            out_classes=1,
            encoder_weights=None
        )
        state_dict = torch.load(MODEL_TO_VALIDATE_PATH, map_location=DEVICE)
        if 'model_state_dict' in state_dict:
            model_to_test.load_state_dict(state_dict['model_state_dict'])
        else:
            model_to_test.load_state_dict(state_dict)

        print("Modelo cargado exitosamente.")
    except Exception as e:
        print(f"Error fatal cargando el modelo: {e}")
        exit()

    average_mae = validate(model_to_test, VALIDATION_DATA_PATH, DEVICE, config)
    print("\n--- Validación Completada ---")
    print(f"Modelo evaluado: {MODEL_TO_VALIDATE_PATH}")
    print(f"MAE Promedio en el conjunto de validación: {average_mae:.2f} m/s")
