import numpy as np
import os
import glob
from tqdm.auto import tqdm
import pandas as pd

OPENFWI_PARENT_PATH = '../open_fwi/' 
REFERENCE_VALUE = 3000.0
ALL_FAMILIES = ['FlatVel', 'CurveVel', 'FlatFault', 'CurveFault', 'Style']

def calculate_family_specific_mae_ref(openfwi_path, target_family, reference_value):
    print(f"--- Iniciando análisis para la familia: {target_family} ---")
    search_pattern = os.path.join(openfwi_path, target_family + '*')
    family_folders = glob.glob(search_pattern)
    if not family_folders:
        print(f"AVISO: No se encontraron carpetas para la familia '{target_family}'. Saltando...")
        return None
    print(f"Carpetas de ground truth encontradas: {', '.join([os.path.basename(f) for f in family_folders])}")
    all_true_velocities = []
    for folder_path in family_folders:
        gt_files = []
        if 'Fault' in target_family:
            gt_files = glob.glob(os.path.join(folder_path, 'vel*.npy'))
        else:
            gt_files = glob.glob(os.path.join(folder_path, 'model', '*.npy'))
        if not gt_files:
            print(f"  - No se encontraron archivos de ground truth para {os.path.basename(folder_path)} con la estructura esperada.")
            continue
        for gt_path in tqdm(gt_files, desc=f"Cargando GT de {os.path.basename(folder_path)}"):
            gt_batch = np.load(gt_path)
            all_true_velocities.extend(gt_batch.flatten())
    if not all_true_velocities:
        print(f"No se pudieron cargar datos para la familia '{target_family}'.")
        return None
    true_velocities_arr = np.array(all_true_velocities)
    mae = np.mean(np.abs(true_velocities_arr - reference_value))
    print(f"MAE de referencia para '{target_family}' calculado: {mae:.4f}")
    return mae

if __name__ == '__main__':
    results = {}
    for family in ALL_FAMILIES:
        mae_ref = calculate_family_specific_mae_ref(
            openfwi_path=OPENFWI_PARENT_PATH,
            target_family=family,
            reference_value=REFERENCE_VALUE
        )
        if mae_ref is not None:
            results[family] = mae_ref
    if results:
        print("\n" + "="*55)
        print(" " * 10 + "RESULTADOS FINALES - MAE DE REFERENCIA")
        print("="*55)
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=['MAE_ref'])
        print(results_df.to_string(float_format="{:.4f}".format))
        print("="*55)