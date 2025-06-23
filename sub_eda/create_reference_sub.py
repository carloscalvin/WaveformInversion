import pandas as pd
import os

PATH_SUBMISSION_ORIGINAL = './submissions/submission_25.6.csv' 
PATH_OUTPUT = './submissions/submission_reference_3000.csv'
VALOR_REFERENCIA = 3000.0

def crear_submission_de_referencia(path_plantilla, path_salida, valor):
    print(f"Cargando plantilla desde: '{path_plantilla}'...")
    if not os.path.exists(path_plantilla):
        print(f"ERROR: El archivo de plantilla no se encontró en '{path_plantilla}'.")
        print("Por favor, asegúrate de que la ruta es correcta y de que has generado un submission primero.")
        return
    try:
        df_plantilla = pd.read_csv(path_plantilla)
        df_referencia = pd.DataFrame()
        df_referencia['oid_ypos'] = df_plantilla['oid_ypos']
        columnas_prediccion = [col for col in df_plantilla.columns if col.startswith('x_')]
        if not columnas_prediccion:
            print("ERROR: No se encontraron columnas de predicción (ej. 'x_1', 'x_3', ...) en el archivo de plantilla.")
            return
        print(f"Se han identificado {len(columnas_prediccion)} columnas de predicción.")
        print(f"Asignando el valor constante '{valor}' a todas las predicciones...")
        for col in columnas_prediccion:
            df_referencia[col] = valor
        os.makedirs(os.path.dirname(path_salida), exist_ok=True)
        df_referencia.to_csv(path_salida, index=False)
        print("\n¡Éxito!")
        print(f"Se ha creado el archivo de submission de referencia en: '{path_salida}'")
        print("Primeras 5 filas del archivo generado:")
        print(df_referencia.head())
    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}")

if __name__ == '__main__':
    crear_submission_de_referencia(
        path_plantilla=PATH_SUBMISSION_ORIGINAL,
        path_salida=PATH_OUTPUT,
        valor=VALOR_REFERENCIA
    )