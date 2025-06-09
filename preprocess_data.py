import os
import numpy as np
from tqdm.auto import tqdm

def create_preprocessed_files(base_data_path, output_path, families_to_use):
    """
    Itera a través de los datos originales, extrae cada muestra individual (mapa + 5 fuentes sísmicas)
    y la guarda en su propio fichero .npz comprimido.

    Args:
        base_data_path (str): Ruta al directorio raíz de los datos de entrenamiento ('kaggle/input/train_samples/').
        output_path (str): Ruta donde se guardarán los ficheros preprocesados ('data/preprocessed_train/').
        families_to_use (list): Lista de las familias de datos a procesar.
    """
    # Crear el directorio de salida principal si no existe
    os.makedirs(output_path, exist_ok=True)
    print(f"Directorio de salida: '{output_path}'")

    for family in tqdm(families_to_use, desc="Procesando Familias"):
        family_path = os.path.join(base_data_path, family)
        output_family_path = os.path.join(output_path, family)
        os.makedirs(output_family_path, exist_ok=True)

        print(f"\nProcesando familia: {family}")

        # Lógica para manejar las diferentes estructuras de directorios del dataset
        is_vel_style = os.path.isdir(os.path.join(family_path, 'model'))
        if is_vel_style:
            data_dir = os.path.join(family_path, 'data')
            model_dir = os.path.join(family_path, 'model')
            file_pairs = [(os.path.join(data_dir, f), os.path.join(model_dir, f.replace('data', 'model'))) for f in os.listdir(data_dir)]
        else: # Estructura de las carpetas de fallas
            vel_files = sorted([f for f in os.listdir(family_path) if f.startswith('vel')])
            file_pairs = [
                (os.path.join(family_path, vf.replace('vel', 'seis')), os.path.join(family_path, vf))
                for vf in vel_files
            ]
        
        for seis_path, vel_path in tqdm(file_pairs, desc="Procesando Ficheros", leave=False):
            try:
                seis_batch = np.load(seis_path)  # Forma: (500, 5, 1000, 70)
                vel_batch = np.load(vel_path)    # Forma: (500, 1, 70, 70)
                
                num_samples = seis_batch.shape[0]

                for i in range(num_samples):
                    # Extraer una única muestra completa
                    velocity_map = vel_batch[i, 0] # Quitamos la dimensión del canal
                    seismic_data_all_sources = seis_batch[i] # Mantenemos las 5 fuentes

                    # Generar un nombre de fichero descriptivo
                    base_filename = os.path.basename(seis_path).replace('.npy', '')
                    output_filename = f"{family}_{base_filename}_sample_{i:03d}.npz"
                    output_filepath = os.path.join(output_family_path, output_filename)
                    
                    # Guardar como .npz comprimido para eficiencia
                    np.savez_compressed(
                        output_filepath, 
                        velocity_map=velocity_map, 
                        seismic_data=seismic_data_all_sources
                    )
            except FileNotFoundError:
                print(f"Aviso: No se encontró el par para {seis_path} o {vel_path}. Saltando.")
            except Exception as e:
                print(f"Error procesando {seis_path}: {e}")

if __name__ == '__main__':
    # --- CONFIGURACIÓN DEL PREPROCESADO ---
    BASE_DATA_PATH = 'kaggle/input/train_samples/'
    OUTPUT_PATH = 'data/preprocessed_train/'
    FAMILIES_TO_USE = [
        'FlatVel_A', 'FlatVel_B',
        'CurveVel_A', 'CurveVel_B',
        'FlatFault_A', 'FlatFault_B',
        'CurveFault_A', 'CurveFault_B',
        'Style_A', 'Style_B'
    ]

    create_preprocessed_files(BASE_DATA_PATH, OUTPUT_PATH, FAMILIES_TO_USE)
    print("\n¡Preprocesamiento completado!")
