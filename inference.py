import os
import torch
import numpy as np
import random
import ps_utils
import utils
import model as model_loader
import torch.nn.functional as F
from tqdm.auto import tqdm

if __name__ == '__main__':
    # --- CONFIGURACIÓN DE INFERENCIA ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAMPLES_TO_VISUALIZE = 12
    MODEL_PATH = 'models/best_unet_model_20250608_175308.pth'
    BASE_DATA_PATH = 'kaggle/input/train_samples/'
    families_to_use = [
        'FlatVel_A', 'FlatVel_B',
        'CurveVel_A', 'CurveVel_B',
        'FlatFault_A', 'FlatFault_B',
        'CurveFault_A', 'CurveFault_B',
        'Style_A', 'Style_B'
    ]

    # Parámetros consistentes con el entrenamiento
    VMIN, VMAX = 1500.0, 4500.0
    VELOCITY_RANGE = VMAX - VMIN
    TARGET_SHAPE = (70, 70)
    DT = 0.001
    NUM_SOURCES_TO_ENSEMBLE = 5

    print(f"Usando dispositivo: {DEVICE}")
    print(f"Cargando modelo desde: {MODEL_PATH}")

    # --- CARGAR MODELO ---
    try:
        best_model = model_loader.SimpleUnet(in_channels=4, out_classes=1)
        best_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        best_model.to(DEVICE)
        best_model.eval()
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        exit()

    # --- INFERENCIA EN MUESTRAS ALEATORIAS ---
    try:
        with torch.no_grad():
            for i in range(SAMPLES_TO_VISUALIZE):        
                random_family = random.choice(families_to_use)
                family_path = os.path.join(BASE_DATA_PATH, random_family)
                is_vel_style = os.path.isdir(os.path.join(family_path, 'model'))
                if is_vel_style:
                    data_dir = os.path.join(family_path, 'data')
                    model_dir = os.path.join(family_path, 'model')
                    seis_filename = random.choice(os.listdir(data_dir))
                    vel_filename = seis_filename.replace('data', 'model')
                    seis_path = os.path.join(data_dir, seis_filename)
                    vel_path = os.path.join(model_dir, vel_filename)
                else:
                    all_vel_files = [f for f in os.listdir(family_path) if f.startswith('vel')]
                    vel_filename = random.choice(all_vel_files)
                    seis_filename = vel_filename.replace('vel', 'seis')
                    seis_path = os.path.join(family_path, seis_filename)
                    vel_path = os.path.join(family_path, vel_filename)

                # Cargar ambos archivos: datos sísmicos y mapa de velocidad real
                seis_file_data = np.load(seis_path)
                vel_file_data = np.load(vel_path)
        
                # Seleccionar una muestra aleatoria de dentro del archivo
                sample_idx = random.randint(0, seis_file_data.shape[0] - 1)
                sample_seismic_data = seis_file_data[sample_idx]
                ground_truth_map = vel_file_data[sample_idx]
                
                if sample_seismic_data.ndim != 3 or sample_seismic_data.shape[0] < NUM_SOURCES_TO_ENSEMBLE:
                    print(f"Aviso: Saltando archivo {seis_path} por tener una forma inesperada: {sample_seismic_data.shape}")
                    continue

                source_predictions = []
                print(f"\nProcesando archivo: {seis_path}")
                
                # Iterar a través de las fuentes para crear el ensembling
                for source_idx in tqdm(range(NUM_SOURCES_TO_ENSEMBLE), desc=f"Fuentes de {seis_path}"):
                    shot_gather = torch.from_numpy(sample_seismic_data[source_idx]).float()
                    # Pre-procesar el sismograma de la fuente actual
                    input_tensor = ps_utils.preprocess_seismic_with_attributes(shot_gather, dt=DT)
                    resized_input = F.interpolate(input_tensor.unsqueeze(0), size=TARGET_SHAPE, mode='bilinear', align_corners=False)

                    # Hacer la predicción para esta fuente
                    prediction_norm = best_model(resized_input.to(DEVICE)).cpu()
                    source_predictions.append(prediction_norm)

                # Apilar y promediar las predicciones de todas las fuentes
                ensembled_prediction_norm = torch.stack(source_predictions).mean(dim=0)
                
                # Desnormalizar el resultado final promediado
                prediction_denorm = ensembled_prediction_norm.squeeze() * VELOCITY_RANGE + VMIN
                ground_truth_denorm = torch.from_numpy(ground_truth_map)

                # Visualizar la predicción final
                utils.plot_map_comparison(
                    map1=ground_truth_denorm[0],
                    title1=f'Mapa Real (Ground Truth)\nFamilia: {random_family}',
                    map2=prediction_denorm,
                    title2=f'Predicción del Modelo (5 Fuentes)\nMuestra: {seis_filename} #{sample_idx}',
                    main_title='Comparación de Verificación: Real vs. Predicción'
                )

    except Exception as e:
        print(f"Ocurrió un error durante la inferencia: {e}")
