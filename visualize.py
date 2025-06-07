import os
import utils
import torch
import numpy as np
import ml_utils
import ps_utils

# Un poco de análisis exploratorio de datos, veamos las ondas sísmicas (posible entrada del modelo)
# y los mapas de velocidad (etiquetas/salida) para la familia Fault primero
path_to_FlatFault_A = 'kaggle/input/train_samples/FlatFault_A'
path_to_CurveFault_A = 'kaggle/input/train_samples/CurveFault_A'

# Buscamos un archivo de velocidad para usarlo como base para flat
files_in_dir = os.listdir(path_to_FlatFault_A)
base_vel_filename = next((f for f in files_in_dir if 'vel' in f), None)

if base_vel_filename:
    seis_filename = base_vel_filename.replace('vel', 'seis')
    vel_file_path = os.path.join(path_to_FlatFault_A, base_vel_filename)
    seis_file_path = os.path.join(path_to_FlatFault_A, seis_filename)

    utils.visualize_sample_pair(
        path_to_vel_file=vel_file_path,
        path_to_seis_file=seis_file_path,
        sample_index=10,
        source_index=2
    )
else:
    print(f"No se encontraron archivos de velocidad ('vel') en la carpeta: {path_to_FlatFault_A}")

# Mostramos otro para curve
files_in_dir = os.listdir(path_to_CurveFault_A)
base_vel_filename = next((f for f in files_in_dir if 'vel' in f), None)

if base_vel_filename:
    seis_filename = base_vel_filename.replace('vel', 'seis')
    vel_file_path = os.path.join(path_to_CurveFault_A, base_vel_filename)
    seis_file_path = os.path.join(path_to_CurveFault_A, seis_filename)

    utils.visualize_sample_pair(
        path_to_vel_file=vel_file_path,
        path_to_seis_file=seis_file_path,
        sample_index=10,
        source_index=2
    )
else:
    print(f"No se encontraron archivos de velocidad ('vel') en la carpeta: {path_to_CurveFault_A}")

# Vamos a visualizar la familia Vel:
path_to_FlatVel_A = 'kaggle/input/train_samples/FlatVel_A'
path_to_CurveVel_A = 'kaggle/input/train_samples/CurveVel_A'

# Buscamos un archivo de velocidad para usarlo como base para flat
vel_folder = os.path.join(path_to_FlatVel_A, 'model')
seis_folder = os.path.join(path_to_FlatVel_A, 'data')
vel_files = sorted(os.listdir(vel_folder))
seis_files = sorted(os.listdir(seis_folder))

if vel_files and seis_files:
    vel_file_path = os.path.join(vel_folder, vel_files[0])
    seis_file_path = os.path.join(seis_folder, seis_files[0])

    utils.visualize_sample_pair(
        path_to_vel_file=vel_file_path,
        path_to_seis_file=seis_file_path,
        sample_index=15,
        source_index=3
    )
else:
    print(f"No se encontraron archivos en las carpetas 'data' o 'model' de {path_to_FlatVel_A}")

# Mostramos otro para curve
vel_folder = os.path.join(path_to_CurveVel_A, 'model')
seis_folder = os.path.join(path_to_CurveVel_A, 'data')
vel_files = sorted(os.listdir(vel_folder))
seis_files = sorted(os.listdir(seis_folder))

if vel_files and seis_files:
    vel_file_path = os.path.join(vel_folder, vel_files[0])
    seis_file_path = os.path.join(seis_folder, seis_files[0])

    utils.visualize_sample_pair(
        path_to_vel_file=vel_file_path,
        path_to_seis_file=seis_file_path,
        sample_index=15,
        source_index=3
    )
else:
    print(f"No se encontraron archivos en las carpetas 'data' o 'model' de {path_to_CurveVel_A}")

# Por último vamos a visualizar la familia Style:
path_to_Style_A = 'kaggle/input/train_samples/Style_A'

vel_folder = os.path.join(path_to_Style_A, 'model')
seis_folder = os.path.join(path_to_Style_A, 'data')
vel_files = sorted(os.listdir(vel_folder))
seis_files = sorted(os.listdir(seis_folder))

if vel_files and seis_files:
    vel_file_path = os.path.join(vel_folder, vel_files[0])
    seis_file_path = os.path.join(seis_folder, seis_files[0])

    utils.visualize_sample_pair(
        path_to_vel_file=vel_file_path,
        path_to_seis_file=seis_file_path,
        sample_index=15,
        source_index=3
    )
else:
    print(f"No se encontraron archivos en las carpetas 'data' o 'model' de {path_to_Style_A}")

# Una baseline, simplemente generando predicciones aleatorios y revisando el MAE
# respecto a un cálculo simple sobre un lote de datos de validación
print("--- Iniciando prueba de baseline con predicciones aleatorias y cálculo simple ---")

# 2. Cargar un lote de datos de validación (usaremos la familia más simple FlatVel)
vel_folder = os.path.join(path_to_FlatVel_A, 'model')
seis_folder = os.path.join(path_to_FlatVel_A, 'data')

vel_files = sorted(os.listdir(vel_folder))
seis_files = sorted(os.listdir(seis_folder))

validation_vel_path = os.path.join(vel_folder, vel_files[0])
validation_seis_path = os.path.join(seis_folder, seis_files[0])

# Cargamos los datos como tensores de PyTorch
y_true = torch.from_numpy(np.load(validation_vel_path)).float()
x_input = torch.from_numpy(np.load(validation_seis_path)).float()

# --- Baseline 1: predicción aleatoria ---
random_preds = torch.rand_like(y_true)
min_vel, max_vel = torch.min(y_true), torch.max(y_true)
scaled_random_preds = random_preds * (max_vel - min_vel) + min_vel
mae_random = ml_utils.calculate_mae(scaled_random_preds, y_true)

# --- Baseline 2: predicción física simple ---
# dt: 1000 pasos en el tiempo, si la simulación es de 1s, dt=0.001
# dx: 70 receptores, si cubren 700m, dx=10
simple_physics_preds = ps_utils.generate_simple_2layer_model(x_input, dt=0.001, dx=10)
mae_physics = ml_utils.calculate_mae(simple_physics_preds, y_true)

# --- Resultados ---
print("\n" + "="*50)
print("      RESULTADOS DE LOS BASELINES")
print("="*50)
print(f"  - MAE con Predicción ALEATORIA: {mae_random.item():.2f}")
print(f"  - MAE con Predicción FÍSICA SIMPLE (2 capas): {mae_physics.item():.2f}")
print("\nEl objetivo es que el MAE físico sea notablemente más bajo que el aleatorio.")

sample_to_visualize = 15

# Comparación 1: Real vs. aleatoria
utils.plot_map_comparison(
    map1=y_true[sample_to_visualize, 0],
    title1=f'Mapa real (muestra #{sample_to_visualize})',
    map2=scaled_random_preds[sample_to_visualize, 0],
    title2=f'Predicción aleatoria\nMAE: {mae_random.item():.2f}',
    main_title='Comparación: ground truth vs. predicción aleatoria'
)

# Comparación 2: Real vs. física simple
utils.plot_map_comparison(
    map1=y_true[sample_to_visualize, 0],
    title1=f'Mapa real (muestra #{sample_to_visualize})',
    map2=simple_physics_preds[sample_to_visualize, 0],
    title2=f'Predicción física simple\nMAE: {mae_physics.item():.2f}',
    main_title='Comparación: ground truth vs. predicción física simple'
)

import matplotlib.pyplot as plt
import time

# --- PRUEBA DE PRE-PROCESADO USANDO BRUGES (probamos familia Vel) ---
print("--- Probando la función de pre-procesado con atributos sísmicos (Vel) ---")

seis_folder = os.path.join(path_to_FlatVel_A, 'data')
seis_files = sorted(os.listdir(seis_folder))
seis_path = os.path.join(seis_folder, seis_files[0])

# Cargamos los datos como tensores de PyTorch
x_input = torch.from_numpy(np.load(seis_path)).float()

sample_index = 20
source_index = 3
single_shot_gather = x_input[sample_index, source_index]

start_time = time.time()
processed_tensor = ps_utils.preprocess_seismic_with_attributes(single_shot_gather)
end_time = time.time()
duration_ms = (end_time - start_time) * 1000

print(f"\nForma del tensor original: {single_shot_gather.shape}")
print(f"Forma del tensor pre-procesado con atributos: {processed_tensor.shape}")
print(f"\n-------------------------------------------------------------")
print(f"  Tiempo de pre-procesado para una muestra: {duration_ms:.2f} ms")
print(f"-------------------------------------------------------------")

channel_titles = [
    '1. Amplitud original (normalizada)',
    '2. Envolvente (energía)',
    '3. Fase instantánea (continuidad)',
    '4. Frecuencia instantánea'
]

fig, axes = plt.subplots(1, 4, figsize=(24, 8))
fig.suptitle(f'Visualización de atributos sísmicos (muestra #{sample_index}, fuente #{source_index})', fontsize=16)

for i, ax in enumerate(axes):
    cmap = 'hsv' if 'Fase' in channel_titles[i] else 'viridis'
    im = ax.imshow(processed_tensor[i].cpu().numpy(), cmap=cmap, aspect='auto')
    ax.set_title(channel_titles[i])
    ax.set_xlabel('Receptor')
    if i == 0:
        ax.set_ylabel('Tiempo')
    fig.colorbar(im, ax=ax, shrink=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

import matplotlib.pyplot as plt

# --- PRUEBA DE PRE-PROCESADO USANDO BRUGES (probamos familia Style) ---
print("--- Probando la función de pre-procesado con atributos sísmicos (Style) ---")

seis_folder = os.path.join(path_to_Style_A, 'data')
seis_files = sorted(os.listdir(seis_folder))
seis_path = os.path.join(seis_folder, seis_files[0])

# Cargamos los datos como tensores de PyTorch
x_input = torch.from_numpy(np.load(seis_path)).float()

sample_index = 20
source_index = 3
single_shot_gather = x_input[sample_index, source_index]

processed_tensor = ps_utils.preprocess_seismic_with_attributes(single_shot_gather)
print(f"\nForma del tensor original: {single_shot_gather.shape}")
print(f"Forma del tensor pre-procesado con atributos: {processed_tensor.shape}")

channel_titles = [
    '1. Amplitud original (normalizada)',
    '2. Envolvente (energía)',
    '3. Fase instantánea (continuidad)',
    '4. Frecuencia instantánea'
]

fig, axes = plt.subplots(1, 4, figsize=(24, 8))
fig.suptitle(f'Visualización de atributos sísmicos (muestra #{sample_index}, fuente #{source_index})', fontsize=16)

for i, ax in enumerate(axes):
    cmap = 'hsv' if 'Fase' in channel_titles[i] else 'viridis'
    im = ax.imshow(processed_tensor[i].cpu().numpy(), cmap=cmap, aspect='auto')
    ax.set_title(channel_titles[i])
    ax.set_xlabel('Receptor')
    if i == 0:
        ax.set_ylabel('Tiempo')
    fig.colorbar(im, ax=ax, shrink=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

import matplotlib.pyplot as plt

# --- PRUEBA DE PRE-PROCESADO USANDO BRUGES (probamos familia Fault) ---
print("--- Probando la función de pre-procesado con atributos sísmicos (Fault) ---")

files_in_dir = os.listdir(path_to_FlatFault_A)
base_seis_path = next((f for f in files_in_dir if 'seis' in f), None)
seis_path = os.path.join(path_to_FlatFault_A, base_seis_path)

# Cargamos los datos como tensores de PyTorch
x_input = torch.from_numpy(np.load(seis_path)).float()

sample_index = 20
source_index = 3
single_shot_gather = x_input[sample_index, source_index]

processed_tensor = ps_utils.preprocess_seismic_with_attributes(single_shot_gather)
print(f"\nForma del tensor original: {single_shot_gather.shape}")
print(f"Forma del tensor pre-procesado con atributos: {processed_tensor.shape}")

channel_titles = [
    '1. Amplitud original (normalizada)',
    '2. Envolvente (energía)',
    '3. Fase instantánea (continuidad)',
    '4. Frecuencia instantánea'
]

fig, axes = plt.subplots(1, 4, figsize=(24, 8))
fig.suptitle(f'Visualización de atributos sísmicos (muestra #{sample_index}, fuente #{source_index})', fontsize=16)

for i, ax in enumerate(axes):
    cmap = 'hsv' if 'Fase' in channel_titles[i] else 'viridis'
    im = ax.imshow(processed_tensor[i].cpu().numpy(), cmap=cmap, aspect='auto')
    ax.set_title(channel_titles[i])
    ax.set_xlabel('Receptor')
    if i == 0:
        ax.set_ylabel('Tiempo')
    fig.colorbar(im, ax=ax, shrink=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
