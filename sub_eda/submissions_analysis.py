import pandas as pd
import os

PATH_CSV = './submissions/test_set_family_predictions.csv'
CONFIDENCE_THRESHOLD = 0.95

def analizar_predicciones(path_csv, umbral_confianza):
    print(f"Cargando y analizando el archivo: '{path_csv}'\n")
    if not os.path.exists(path_csv):
        print(f"ERROR: El archivo no se encontró en la ruta especificada: '{path_csv}'")
        return
    try:
        df = pd.read_csv(path_csv)
        required_cols = ['predicted_family', 'confidence']
        if not all(col in df.columns for col in required_cols):
            print(f"ERROR: El CSV debe contener las columnas: {required_cols}")
            return
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        return

    print("--- ANÁLISIS 1: Distribución general de familias ---")
    print("Muestra la cantidad y el porcentaje de cada familia en todo el test set.")

    counts = df['predicted_family'].value_counts()
    percentages = df['predicted_family'].value_counts(normalize=True)

    distribucion_general = pd.DataFrame({'conteo': counts, 'porcentaje': percentages})
    print(distribucion_general)
    print("\n" + "="*60 + "\n")

    print(f"--- ANÁLISIS 2: Distribución para confianza > {umbral_confianza:.0%} ---")
    print("Muestra la distribución solo para las predicciones más seguras del modelo.")

    df_alta_confianza = df[df['confidence'] > umbral_confianza]
    if df_alta_confianza.empty:
        print(f"AVISO: No se encontraron predicciones con una confianza superior a {umbral_confianza:.0%}.")
    else:
        counts_conf = df_alta_confianza['predicted_family'].value_counts()
        percentages_conf = df_alta_confianza['predicted_family'].value_counts(normalize=True)
        distribucion_conf = pd.DataFrame({'conteo': counts_conf, 'porcentaje': percentages_conf})
        print(f"Total de muestras con alta confianza: {len(df_alta_confianza)} de {len(df)} ({len(df_alta_confianza)/len(df):.2%})")
        print(distribucion_conf)
    print("\n" + "="*60 + "\n")

    print("--- ANÁLISIS 3: Análisis de las puntuaciones de confianza ---")
    print("Muestra estadísticas descriptivas (media, mediana, desviación, etc.) de la confianza.")
    
    print("\n[ Confianza general (todas las predicciones) ]")
    print(df['confidence'].describe().apply("{:.4f}".format))
    print("\n[ Confianza por familia ]")
    confianza_por_clase = df.groupby('predicted_family')['confidence'].describe()
    print(confianza_por_clase.to_string(float_format="{:.4f}".format))

if __name__ == '__main__':
    analizar_predicciones(
        path_csv=PATH_CSV,
        umbral_confianza=CONFIDENCE_THRESHOLD
    )