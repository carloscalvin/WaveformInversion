import os
import torch
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
import model as model_loader
import ps_utils

def calculate_mae_denorm(prediction_denorm, ground_truth_denorm):
    return torch.mean(torch.abs(prediction_denorm - ground_truth_denorm)).item()

def validate_per_family(model, validation_data_path, device, config):
    model.to(device)
    model.eval()

    family_mae_scores = defaultdict(list)
    
    try:
        families_to_validate = [d for d in os.listdir(validation_data_path) if os.path.isdir(os.path.join(validation_data_path, d))]
        if not families_to_validate:
            print(f"Advertencia: No se encontraron directorios de familias en '{validation_data_path}'")
            return {'overall': float('inf')}
        print(f"Familias encontradas para validar: {families_to_validate}")
    except FileNotFoundError:
        print(f"Error: El directorio de validación no existe: '{validation_data_path}'")
        return {'overall': float('inf')}

    with torch.no_grad():
        for family in tqdm(families_to_validate, desc="Validando Familias"):
            family_path = os.path.join(validation_data_path, family)
            
            is_vel_style = os.path.isdir(os.path.join(family_path, 'model'))
            if is_vel_style:
                data_dir = os.path.join(family_path, 'data')
                seis_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.npy', '.npz'))]
            else:
                all_files = os.listdir(family_path)
                vel_files = [f for f in all_files if f.startswith('vel') and f.endswith(('.npy', '.npz'))]
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

                        family_mae_scores[family].append(mae)

                except Exception as e:
                    print(f"Error procesando {seis_path} en el índice {sample_idx}: {e}")
                    continue
    
    final_results = {}
    all_scores_for_overall_avg = []
    
    print("\n" + "="*40)
    print("  Resultados de MAE por Familia")
    print("="*40)

    for family, scores in sorted(family_mae_scores.items()):
        if scores:
            mean_mae = np.mean(scores)
            final_results[family] = mean_mae
            all_scores_for_overall_avg.extend(scores)
            print(f"- {family:<15}: {mean_mae:.2f} m/s  ({len(scores)} muestras)")
    
    if not all_scores_for_overall_avg:
        print("\nNo se pudo calcular ningún MAE. Revisa las rutas y el formato de los datos.")
        final_results['overall'] = float('inf')
    else:
        overall_mae = np.mean(all_scores_for_overall_avg)
        final_results['overall'] = overall_mae
        print("----------------------------------------")
        print(f"- MAE Promedio General: {overall_mae:.2f} m/s")
        
    return final_results

if __name__ == '__main__':
    MODEL_TO_VALIDATE_PATH = 'models/best_efficientnetb7_model_20250621_173049.pth'
    VALIDATION_DATA_PATH = 'open_fwi'
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

    results = validate_per_family(model_to_test, VALIDATION_DATA_PATH, DEVICE, config)
    
    print("\n--- Validación Finalizada ---")
