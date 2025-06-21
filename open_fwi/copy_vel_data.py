import os
import shutil

# Parámetros
FAMILIES_TO_USE = [
    'FlatVel_A', 'FlatVel_B',
    'CurveVel_A', 'CurveVel_B',
    'Style_A', 'Style_B'
]

start_file = 21
num_files_to_copy = 2

# Ruta base
drive_base_path = r"G:\Mi unidad\WaveformInversionProject\OpenFWI"
dataset_base_path = "./dataset"

# Función para copiar ficheros
def copy_family_files(family, start_num, num_files):
    data_src_folder = f"{drive_base_path}/{family}/data"
    model_src_folder = f"{drive_base_path}/{family}/model"
    data_dst_folder = f"{dataset_base_path}/{family}/data"
    model_dst_folder = f"{dataset_base_path}/{family}/model"

    os.makedirs(data_dst_folder, exist_ok=True)
    os.makedirs(model_dst_folder, exist_ok=True)

    for i in range(start_num, start_num + num_files):
        data_src = f"{data_src_folder}/data{i}.npy"
        model_src = f"{model_src_folder}/model{i}.npy"
        data_dst = f"{data_dst_folder}/data{i}.npy"
        model_dst = f"{model_dst_folder}/model{i}.npy"

        try:
            shutil.copy(data_src, data_dst)
            shutil.copy(model_src, model_dst)
            print(f"[{family}] Copiado data{i}.npy y model{i}.npy")
        except FileNotFoundError as e:
            print(f"[{family}] Archivo faltante: {e.filename}")

# Ejecutar para cada familia
for family in FAMILIES_TO_USE:
    copy_family_files(family, start_file, num_files_to_copy)
