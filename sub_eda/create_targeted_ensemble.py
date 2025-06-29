import pandas as pd
import os

PATH_MODEL_A = './submissions/submission_style18.csv' 
PATH_MODEL_B = './submissions/ensemble_flatvel_from_submission_28.2_to_25.6.csv'
PATH_CLASSIFICATION = './submissions/test_set_family_predictions.csv'
PATH_OUTPUT = './submissions/ensemble_final_epoch18.csv'
TARGET_FAMILY = 'Style'
CONFIDENCE_THRESHOLD = 0.98
WEIGHT_A = 1
WEIGHT_B = 0

def create_targeted_ensemble(path_a, path_b, classification_path, output_path,
                             target_family, confidence_threshold, weight_a, weight_b):
    print("Iniciando la creación de un submission de ensamble dirigido...")
    assert round(weight_a + weight_b, 5) == 1.0, "Los pesos (weights) deben sumar 1.0"
    for path in [path_a, path_b, classification_path]:
        if not os.path.exists(path):
            print(f"ERROR: No se encontró el archivo de entrada: '{path}'")
            return
    
    try:
        print(f"Cargando modelo A desde: '{path_a}'")
        df_a = pd.read_csv(path_a)
        df_a = df_a.sort_values(by='oid_ypos').reset_index(drop=True)

        print(f"Cargando modelo B desde: '{path_b}'")
        df_b = pd.read_csv(path_b)
        df_b = df_b.sort_values(by='oid_ypos').reset_index(drop=True)
        
        print(f"Cargando clasificación de familias desde: '{classification_path}'")
        df_classification = pd.read_csv(classification_path)

        if not df_a.shape == df_b.shape or not all(df_a['oid_ypos'] == df_b['oid_ypos']):
            print("ERROR: Los archivos de submission A y B no tienen la misma forma o no están ordenados igual.")
            return

        print(f"Filtrando muestras para la familia '{target_family}' con confianza > {confidence_threshold:.0%}")

        family_samples = df_classification[df_classification['predicted_family'] == target_family]
        high_confidence_samples = family_samples[family_samples['confidence'] > confidence_threshold]
        target_oids = high_confidence_samples['oid'].unique()
        
        if len(target_oids) == 0:
            print("AVISO: No se encontraron muestras que cumplan los criterios de familia y confianza.")
            return

        print(f"Se han seleccionado {len(target_oids)} muestras ('oid') para el ensamble.")

        df_ensemble = df_b.copy()

        rows_to_ensemble_mask = df_ensemble['oid_ypos'].str.split('_').str[0].isin(target_oids)
        num_rows = rows_to_ensemble_mask.sum()
        print(f"Se aplicará el ensamble a {num_rows} filas.")

        prediction_columns = [col for col in df_b.columns if col.startswith('x_')]

        print(f"Aplicando ensamble con pesos: modelo A ({weight_a*100}%) y modelo B ({weight_b*100}%)")
        df_ensemble.loc[rows_to_ensemble_mask, prediction_columns] = \
            (weight_a * df_a.loc[rows_to_ensemble_mask, prediction_columns]) + \
            (weight_b * df_b.loc[rows_to_ensemble_mask, prediction_columns])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_ensemble.to_csv(output_path, index=False)

        print("\n¡Éxito!")
        print(f"Archivo de ensamble guardado en: '{output_path}'")

    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}")

if __name__ == '__main__':
    create_targeted_ensemble(
        path_a=PATH_MODEL_A,
        path_b=PATH_MODEL_B,
        classification_path=PATH_CLASSIFICATION,
        output_path=PATH_OUTPUT,
        target_family=TARGET_FAMILY,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        weight_a=WEIGHT_A,
        weight_b=WEIGHT_B
    )