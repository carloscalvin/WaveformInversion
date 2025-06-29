import pandas as pd
import os

BASE_SUBMISSION_PATH = './submissions/submission_style18.csv' 
FAMILY_CLASSIFICATION_PATH = './submissions/test_set_family_predictions.csv'
OUTPUT_PATH = './submissions/submission_style18_style_to_3000.csv'
TARGET_FAMILY = 'Style'
TARGET_VALUE = 3000.0

def create_experimental_submission(base_path, classification_path, output_path, target_family, value):
    for path in [base_path, classification_path]:
        if not os.path.exists(path):
            print(f"ERROR: No se encontró el archivo de entrada: '{path}'")
            return
    try:
        print(f"Cargando submission base desde: '{base_path}'")
        submission_df = pd.read_csv(base_path)

        print(f"Cargando clasificación de familias desde: '{classification_path}'")
        families_df = pd.read_csv(classification_path)
        
        print(f"Identificando muestras de la familia: '{target_family}'...")
        oids_to_modify = families_df[families_df['predicted_family'] == target_family]['oid'].unique()
        
        if len(oids_to_modify) == 0:
            print(f"AVISO: No se encontró ninguna muestra para la familia '{target_family}' en el archivo de clasificación.")
            return
            
        print(f"Se encontraron {len(oids_to_modify)} muestras ('oid') pertenecientes a la familia '{target_family}'.")

        rows_to_modify_mask = submission_df['oid_ypos'].str.split('_').str[0].isin(oids_to_modify)
        num_rows = rows_to_modify_mask.sum()
        if num_rows == 0:
            print("ERROR: Se encontraron los 'oid' pero no se pudo encontrar ninguna fila coincidente en el archivo de submission.")
            return

        print(f"Se modificarán {num_rows} filas en el archivo de submission.")
        
        print(f"Asignando el valor '{value}' a las predicciones de estas filas...")
        prediction_columns = [col for col in submission_df.columns if col.startswith('x_')]
        submission_df.loc[rows_to_modify_mask, prediction_columns] = value
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        submission_df.to_csv(output_path, index=False)
        
        print("\n¡Éxito!")
        print(f"Archivo de submission experimental guardado en: '{output_path}'")
        
        print("\nVerificando los primeros valores modificados:")
        print(submission_df[rows_to_modify_mask].head())

    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}")

if __name__ == '__main__':
    create_experimental_submission(
        base_path=BASE_SUBMISSION_PATH,
        classification_path=FAMILY_CLASSIFICATION_PATH,
        output_path=OUTPUT_PATH,
        target_family=TARGET_FAMILY,
        value=TARGET_VALUE
    )