import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm.auto import tqdm
import model as model_loader 
import ps_utils

if __name__ == '__main__':
    config = {
        'GENERALIST_MODEL_PATH': '../models/best_efficientnetb7_model_20250621_173049.pth',
        'CLASSIFIER_MODEL_PATH': '../classifier/models/best_classifier_model.pth',
        'TEST_DATA_PATH': '../kaggle/input/test/',
        'OUTPUT_CSV_PATH': './submissions/test_set_family_predictions.csv',
        'DEVICE': "cuda" if torch.cuda.is_available() else "cpu",
        'VMIN': 1500.0, 'VMAX': 4500.0, 'DT': 0.001, 'NUM_SOURCES_TO_ENSEMBLE': 5
    }
    config['VELOCITY_RANGE'] = config['VMAX'] - config['VMIN']

    print(f"Usando dispositivo: {config['DEVICE']}")
    os.makedirs(os.path.dirname(config['OUTPUT_CSV_PATH']), exist_ok=True)
    
    print(f"Cargando modelo generalista: {config['GENERALIST_MODEL_PATH']}")
    generalist_model = model_loader.SimpleUnet(encoder_name="timm-efficientnet-b7", 
                                               in_channels=4, out_classes=1, 
                                               encoder_weights=None)
    generalist_model.load_state_dict(torch.load(config['GENERALIST_MODEL_PATH'], map_location=config['DEVICE']))
    
    print(f"Cargando modelo clasificador: {config['CLASSIFIER_MODEL_PATH']}")
    class_names = ['CurveFault', 'CurveVel', 'FlatFault', 'FlatVel', 'Style']
    classifier_model = models.resnet18(weights=None)
    classifier_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    classifier_model.fc = nn.Linear(classifier_model.fc.in_features, len(class_names))
    classifier_model.load_state_dict(torch.load(config['CLASSIFIER_MODEL_PATH'], map_location=config['DEVICE']))
    classifier_model.to(config['DEVICE']).eval()
    test_files = [f for f in os.listdir(config['TEST_DATA_PATH']) if f.endswith('.npy')]
    family_predictions = []

    print(f"\nProcesando y clasificando {len(test_files)} muestras del test set...")
    for filename in tqdm(test_files, desc="Procesando test set"):
        oid = filename.replace('.npy', '')
        seis_path = os.path.join(config['TEST_DATA_PATH'], filename)
        seismic_data_sample = torch.from_numpy(np.load(seis_path)).float()
        our_predicted_map = ps_utils.generate_map_from_seismic(generalist_model, 
                                                               seismic_data_sample,
                                                               config['DEVICE'],
                                                               config['NUM_SOURCES_TO_ENSEMBLE'],
                                                               config['DT'],
                                                               config['VELOCITY_RANGE'],
                                                               config['VMIN'])
        norm_map = (torch.from_numpy(our_predicted_map) - config['VMIN']) / (config['VELOCITY_RANGE'])
        input_tensor = norm_map.unsqueeze(0).unsqueeze(0).to(config['DEVICE'])
        with torch.no_grad():
            output = classifier_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)
            predicted_family = class_names[pred_idx.item()]
        family_predictions.append({
            'oid': oid, 
            'predicted_family': predicted_family,
            'confidence': confidence.item()
        })

    results_df = pd.DataFrame(family_predictions)
    results_df.to_csv(config['OUTPUT_CSV_PATH'], index=False)
    print(f"\nPredicciones de familias guardadas en: {config['OUTPUT_CSV_PATH']}")
    print("\n--- Distribución de familias estimada en el test set ---")
    print(results_df['predicted_family'].value_counts(normalize=True).apply("{:.2%}".format))
