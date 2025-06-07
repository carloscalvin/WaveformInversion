import os
import torch
import numpy as np
import random
import ps_utils
import utils
import model as model_loader
import torch.nn.functional as F

if __name__ == '__main__':
    # --- CONFIGURACIÓN DE INFERENCIA ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = 'models/best_unet_model_20250607_232020.pth'
    
    # Ruta a los datos de test
    PATH_TO_TEST_DATA = 'kaggle/input/test/'
    
    # Parámetros consistentes con el entrenamiento
    VMIN, VMAX = 1500.0, 4500.0
    VELOCITY_RANGE = VMAX - VMIN
    TARGET_SHAPE = (70, 70)
    DT = 0.001

    print(f"Usando dispositivo: {DEVICE}")
    print(f"Cargando modelo desde: {MODEL_PATH}")

    # --- CARGAR MODELO ---
    try:
        best_model = model_loader.SimpleUnet(in_channels=4, out_classes=1)
        best_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        best_model.to(DEVICE)
        best_model.eval()
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        exit()

    # --- INFERENCIA EN MUESTRAS ALEATORIAS ---
    try:
        test_files = [f for f in os.listdir(PATH_TO_TEST_DATA) if f.endswith('.npy')]
        # Seleccionar 5 archivos al azar
        random_files_to_test = random.sample(test_files, min(5, len(test_files)))

        print(f"\nRealizando inferencia en {len(random_files_to_test)} archivos de test aleatorios...")

        with torch.no_grad():
            for filename in random_files_to_test:
                # Cargar y pre-procesar una muestra aleatoria del archivo
                seis_path = os.path.join(PATH_TO_TEST_DATA, filename)
                seis_batch = np.load(seis_path)
                
                sample_idx = random.randint(0, seis_batch.shape[0] - 1)
                shot_gather = torch.from_numpy(seis_batch[sample_idx]).float()
                
                input_tensor = ps_utils.preprocess_seismic_with_attributes(shot_gather, dt=DT)
                resized_input = F.interpolate(input_tensor.unsqueeze(0), size=TARGET_SHAPE, mode='bilinear', align_corners=False)

                # Hacer la predicción
                prediction_norm = best_model(resized_input.to(DEVICE)).cpu()
                
                # Desnormalizar
                prediction_denorm = prediction_norm.squeeze() * VELOCITY_RANGE + VMIN

                # Visualizar
                utils.plot_inference_result(
                    prediction_denorm.numpy(),
                    title=f'Predicción para Muestra Aleatoria de:\n{filename}'
                )

    except Exception as e:
        print(f"Ocurrió un error durante la inferencia: {e}")
