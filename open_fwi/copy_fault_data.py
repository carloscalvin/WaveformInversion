import os
import shutil

# Configuración de familias y prefijos según tipo A o B
family_prefixes = {
    'FlatFault_A': ['seis2', 'seis4', 'vel2', 'vel4'],
    'FlatFault_B': ['seis6', 'seis8', 'vel6', 'vel8'],
    'CurveFault_A': ['seis2', 'seis4', 'vel2', 'vel4'],
    'CurveFault_B': ['seis6', 'seis8', 'vel6', 'vel8'],
}

# Número de archivos por prefijo
start_index = 11
num_files_per_prefix = 1

# Ruta base
drive_base_path = r"G:\Mi unidad\WaveformInversionProject\OpenFWI"
dataset_base_path = "./dataset/open_fwi"

# Función para copiar archivos según prefijos
def copy_fault_family_files(family, prefixes, start_idx, num_files):
    dst_folder = os.path.join(dataset_base_path, family)
    os.makedirs(dst_folder, exist_ok=True)

    for prefix in prefixes:
        for i in range(start_idx, start_idx + num_files):
            filename = f"{prefix}_1_{i}.npy"
            src = os.path.join(drive_base_path, family, filename)
            dst = os.path.join(dst_folder, filename)

            try:
                shutil.copy(src, dst)
                print(f"[{family}] Copiado {filename}")
            except FileNotFoundError:
                print(f"[{family}] Archivo faltante: {filename}")

# Ejecutar para cada familia
for family, prefixes in family_prefixes.items():
    copy_fault_family_files(family, prefixes, start_index, num_files_per_prefix)
