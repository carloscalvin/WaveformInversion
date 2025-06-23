import pandas as pd
import numpy as np
import os
import torch
import glob
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import math
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import model as model_loader 
import ps_utils

GENERALIST_MODEL_PATH = '../models/best_efficientnetb7_model_20250621_173049.pth'
KAGGLE_DATA_PATH = '../kaggle/input/test/'
KAGGLE_CLASSIFICATION_CSV = './submissions/test_set_family_predictions.csv'
OPENFWI_PARENT_PATH = '../open_fwi/'
TARGET_FAMILY = 'Style'
TOTAL_SAMPLES_PER_DATASET = 1000

config = {
    'DEVICE': "cuda" if torch.cuda.is_available() else "cpu",
    'VMIN': 1500.0, 'VMAX': 4500.0, 'DT': 0.001, 'NUM_SOURCES_TO_ENSEMBLE': 5
}
config['VELOCITY_RANGE'] = config['VMAX'] - config['VMIN']

def get_velocities_from_preclassified_set(target_family, classification_csv_path, data_path, gen_model, config, total_samples):
    print(f"Procesando el dataset de Kaggle para la familia '{target_family}'...")
    try:
        df_classification = pd.read_csv(classification_csv_path)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de clasificación en {classification_csv_path}")
        return []

    target_oids = df_classification[df_classification['predicted_family'] == target_family]['oid'].tolist()[:total_samples]
    
    if not target_oids:
        print(f"No se encontraron OIDs para la familia '{target_family}' en el CSV.")
        return []

    all_velocities = []
    for oid in tqdm(target_oids, desc=f"Generando mapas para Kaggle ({target_family})"):
        seis_path = os.path.join(data_path, f"{oid}.npy")
        if os.path.exists(seis_path):
            seismic_data_sample = torch.from_numpy(np.load(seis_path)).float()
            predicted_map = ps_utils.generate_map_from_seismic(gen_model, 
                                                                seismic_data_sample,
                                                                config['DEVICE'],
                                                                config['NUM_SOURCES_TO_ENSEMBLE'],
                                                                config['DT'],
                                                                config['VELOCITY_RANGE'],
                                                                config['VMIN'])            
            all_velocities.extend(predicted_map.flatten())
            
    return all_velocities

def get_openfwi_velocities_from_folders(target_family, openfwi_path, gen_model, config, total_samples):
    print(f"Procesando un total de {total_samples} muestras de OpenFWI para la familia '{target_family}'...")
    
    search_pattern = os.path.join(openfwi_path, target_family + '*')
    family_folders = glob.glob(search_pattern)

    if not family_folders:
        print(f"AVISO: No se encontraron carpetas para la familia '{target_family}'.")
        return []
    
    num_subfolders = len(family_folders)
    samples_per_folder = math.ceil(total_samples / num_subfolders)
    print(f"Se encontraron {num_subfolders} sub-carpetas. Se intentará coger ~{samples_per_folder} muestras de cada una.")

    all_velocities = []
    for folder_path in family_folders:
        folder_name = os.path.basename(folder_path)
        print(f"Procesando sub-carpeta: {folder_name}")
        
        seismic_files_in_folder = glob.glob(os.path.join(folder_path, 'data', '*.npy'))
        if not seismic_files_in_folder:
            print(f"  - No se encontraron archivos en {folder_path}/data")
            continue

        samples_processed_count = 0
        
        for seis_path in tqdm(seismic_files_in_folder, desc=f"Procesando {folder_name}"):
            if samples_processed_count >= samples_per_folder:
                break

            batch_seismic_data = np.load(seis_path)
            
            for sample_seismic_data in batch_seismic_data:
                if samples_processed_count >= samples_per_folder:
                    break

                seismic_tensor = torch.from_numpy(sample_seismic_data).float()
                predicted_map = ps_utils.generate_map_from_seismic(gen_model, 
                                                                    seismic_tensor,
                                                                    config['DEVICE'],
                                                                    config['NUM_SOURCES_TO_ENSEMBLE'],
                                                                    config['DT'],
                                                                    config['VELOCITY_RANGE'],
                                                                    config['VMIN'])     
                all_velocities.extend(predicted_map.flatten())
                samples_processed_count += 1
    
    return all_velocities

if __name__ == '__main__':
    print("Cargando modelos...")
    print(f"Cargando modelo generalista: {GENERALIST_MODEL_PATH}")
    generalist_model = model_loader.SimpleUnet(encoder_name="timm-efficientnet-b7", 
                                               in_channels=4, out_classes=1, 
                                               encoder_weights=None)
    generalist_model.load_state_dict(torch.load(GENERALIST_MODEL_PATH, map_location=config['DEVICE']))
    print("Modelo cargado.")

    kaggle_velocities = get_velocities_from_preclassified_set(TARGET_FAMILY, KAGGLE_CLASSIFICATION_CSV, KAGGLE_DATA_PATH, 
                                                              generalist_model, config, TOTAL_SAMPLES_PER_DATASET)
    openfwi_velocities = get_openfwi_velocities_from_folders(TARGET_FAMILY, OPENFWI_PARENT_PATH, 
                                                             generalist_model, config, TOTAL_SAMPLES_PER_DATASET)

    if not kaggle_velocities or not openfwi_velocities:
        print(f"\nNo se pudieron obtener las velocidades para la familia '{TARGET_FAMILY}' en uno o ambos datasets. No se puede comparar.")
    else:
        print("\n--- Comparación de estadísticas descriptivas ---")
        kaggle_series = pd.Series(kaggle_velocities)
        openfwi_series = pd.Series(openfwi_velocities)
        
        comparison_df = pd.DataFrame({
            'Kaggle test set': kaggle_series.describe(),
            'OpenFWI set': openfwi_series.describe()
        })
        print(comparison_df.round(2))

        print("\nGenerando histograma comparativo...")
        plt.figure(figsize=(12, 7))
        plt.hist(kaggle_series, bins=100, alpha=0.7, label='Kaggle test set preds', density=True)
        plt.hist(openfwi_series, bins=100, alpha=0.7, label='OpenFWI set preds', density=True)
        plt.title(f"Distribución de velocidades predichas para la familia: {TARGET_FAMILY}", fontsize=16)
        plt.xlabel("Velocidad (m/s)")
        plt.ylabel("Densidad")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()
