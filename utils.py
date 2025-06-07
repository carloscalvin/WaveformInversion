import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

def format_bytes(size_bytes):
    """Convierte un tamaño en bytes a un formato legible (KB, MB, GB, TB)."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = 0
    while size_bytes >= 1024 and i < len(size_name) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f} {size_name[i]}"

def analyze_directory(root_path):
    """
    Analiza un directorio para contar archivos .npy y calcular el tamaño total,
    mostrando un resumen detallado de cada subdirectorio.
    """
    print(f"Iniciando análisis del directorio: '{root_path}'")

    start_time = time.time()
    try:
        if not os.path.isdir(root_path):
            raise FileNotFoundError(f"El directorio raíz no existe. Revisa la ruta: {root_path}")

        total_files = 0
        total_size_bytes = 0
        directory_info = {}

        for dirpath, dirnames, filenames in os.walk(root_path):
            npy_files = [f for f in filenames if f.endswith('.npy')]
            if npy_files:
                num_files_in_dir = len(npy_files)
                size_in_dir_bytes = sum(os.path.getsize(os.path.join(dirpath, f)) for f in npy_files)
                
                relative_path = os.path.relpath(dirpath, os.path.dirname(root_path))
                directory_info[relative_path] = {
                    'num_files': num_files_in_dir,
                    'size_bytes': size_in_dir_bytes
                }
                total_files += num_files_in_dir
                total_size_bytes += size_in_dir_bytes

        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*50)
        print(" ANÁLISIS DEL DIRECTORIO COMPLETADO")
        print("="*50)
        print(f"Tiempo de ejecución: {duration:.2f} segundos")
        print(f"\nResumen General:")
        print(f"  - Número Total de Archivos (.npy): {total_files}")
        print(f"  - Tamaño Total Acumulado: {format_bytes(total_size_bytes)}")
        
        if directory_info:
            print("\nDesglose por Directorio:")
            sorted_dirs = sorted(directory_info.keys())
            for dir_path in sorted_dirs:
                info = directory_info[dir_path]
                print(f"  - Carpeta: {dir_path}")
                print(f"    - Archivos: {info['num_files']}")
                print(f"    - Tamaño:   {format_bytes(info['size_bytes'])}")
        else:
            print("\nNo se encontraron archivos .npy en ningún subdirectorio.")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
    except Exception as e:
        print(f"\nOcurrió un error inesperado: {e}")

def visualize_sample_pair(path_to_vel_file, path_to_seis_file, sample_index=0, source_index=0):
    """
    Visualiza un par de muestras (mapa de velocidad y sismograma) lado a lado,
    recibiendo las rutas completas a los archivos.
    """
    try:
        print(f"Cargando muestra #{sample_index} de:")
        print(f" - Mapa de Velocidad: {path_to_vel_file}")
        print(f" - Datos Sísmicos: {path_to_seis_file}")

        vel_batch = np.load(path_to_vel_file)
        seis_batch = np.load(path_to_seis_file)

        velocity_map = vel_batch[sample_index, 0]
        seismic_shot_gather = seis_batch[sample_index, source_index]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        im1 = ax1.imshow(velocity_map, cmap='viridis')
        fig.colorbar(im1, ax=ax1, label='Velocidad (m/s)', shrink=0.8)
        ax1.set_title(f'Salida: Mapa de Velocidad\n(Muestra #{sample_index})')
        ax1.set_xlabel('Posición X'); ax1.set_ylabel('Profundidad Y')

        vmax = np.percentile(seismic_shot_gather, 99)
        im2 = ax2.imshow(seismic_shot_gather, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
        fig.colorbar(im2, ax=ax2, label='Amplitud de la Onda', shrink=0.8)
        ax2.set_title(f'Entrada: Sismograma\n(Muestra #{sample_index}, Fuente #{source_index})')
        ax2.set_xlabel('Número de Receptor'); ax2.set_ylabel('Pasos de Tiempo')

        plt.tight_layout(); plt.show()

    except FileNotFoundError:
        print("Error: No se encontró uno de los archivos. Revisa las rutas.")
    except Exception as e:
        print(f"Ocurrió un error durante la visualización: {e}")
    
def plot_map_comparison(map1, title1, map2, title2, main_title="Comparación de Mapas de Velocidad"):
    """
    Visualiza dos mapas de velocidad lado a lado con una escala de color compartida.
    Acepta tensores de PyTorch o arrays de NumPy.
    """
    # Convertir a NumPy arrays si son tensores de PyTorch
    if isinstance(map1, torch.Tensor):
        map1 = map1.cpu().numpy()
    if isinstance(map2, torch.Tensor):
        map2 = map2.cpu().numpy()

    # Calcular la escala de color compartida para una comparación justa
    vmin = min(np.min(map1), np.min(map2))
    vmax = max(np.max(map1), np.max(map2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(main_title, fontsize=16)

    # --- Mapa 1 ---
    im1 = ax1.imshow(map1, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title(title1)
    ax1.set_xlabel('Posición X'); ax1.set_ylabel('Profundidad Y')
    fig.colorbar(im1, ax=ax1, label='Velocidad (m/s)', shrink=0.8)

    # --- Mapa 2 ---
    im2 = ax2.imshow(map2, cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title(title2)
    ax2.set_xlabel('Posición X'); ax2.set_ylabel('') # No repetir ylabel
    fig.colorbar(im2, ax=ax2, label='Velocidad (m/s)', shrink=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para el supertítulo
    plt.show()

def plot_inference_result(pred_map, title):
    """Función simple para visualizar una única predicción."""
    plt.figure(figsize=(8, 8))
    plt.imshow(pred_map, cmap='viridis')
    plt.colorbar(label='Velocidad (m/s)', shrink=0.8)
    plt.title(title)
    plt.xlabel('Posición X')
    plt.ylabel('Profundidad Y')
    plt.show()
