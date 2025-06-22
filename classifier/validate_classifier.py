import os
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import model as model_loader 
import ps_utils

config = {
    'GENERALIST_MODEL_PATH': '../models/best_efficientnetb7_model_20250621_173049.pth',
    'CLASSIFIER_MODEL_PATH': './models/best_classifier_model.pth',
    'VALIDATION_DATA_PATH': '../open_fwi/',
    'DEVICE': "cuda" if torch.cuda.is_available() else "cpu",
    'VMIN': 1500.0,
    'VMAX': 4500.0,
    'DT': 0.001,
    'NUM_SOURCES_TO_ENSEMBLE': 5,
}
config['VELOCITY_RANGE'] = config['VMAX'] - config['VMIN']

def get_superfamily_info(root_dir):
    print("Escaneando directorios para identificar súper-familias...")
    try:
        variant_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    except FileNotFoundError:
        print(f"ERROR: El directorio raíz no existe: {root_dir}")
        return None, None
    
    super_family_names = set(d.rsplit('_', 1)[0] for d in variant_dirs)
    class_names = sorted(list(super_family_names))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    
    print(f"Encontradas {len(class_names)} súper-familias: {class_names}")
    return class_names, class_to_idx

def validate_end_to_end(generalist_model, classifier_model, validation_path, device, config, class_to_idx):
    generalist_model.to(device).eval()
    classifier_model.to(device).eval()

    all_preds = []
    all_labels = []
    
    try:
        families_to_validate = [d for d in os.listdir(validation_path) if os.path.isdir(os.path.join(validation_path, d))]
        if not families_to_validate:
            print(f"Advertencia: No se encontraron directorios de familias en '{validation_path}'")
            return [], []
    except FileNotFoundError:
        print(f"Error: El directorio de validación no existe: '{validation_path}'")
        return [], []

    with torch.no_grad():
        for family_variant in tqdm(families_to_validate, desc="Validando súper-familias"):
            family_path = os.path.join(validation_path, family_variant)
            super_family_name = family_variant.rsplit('_', 1)[0]
            true_label_idx = class_to_idx[super_family_name]
            is_vel_style = os.path.isdir(os.path.join(family_path, 'model'))
            if is_vel_style:
                data_dir = os.path.join(family_path, 'data')
                if not os.path.isdir(data_dir): continue
                seis_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.npy', '.npz'))]
            else:
                all_files = os.listdir(family_path)
                vel_files = [f for f in all_files if f.startswith('vel') and f.endswith(('.npy', '.npz'))]
                seis_files = [os.path.join(family_path, v.replace('vel', 'seis')) for v in vel_files]
            for seis_path in tqdm(seis_files, desc=f"Ficheros en {family_variant}", leave=False):
                try:
                    if not os.path.exists(seis_path):
                        print(f"Advertencia: No se encontró el archivo sísmico correspondiente: {seis_path}")
                        continue
                    seis_file_data = np.load(seis_path)
                    for sample_idx in range(seis_file_data.shape[0]):
                        sample_seismic_data = seis_file_data[sample_idx]
                        if sample_seismic_data.ndim != 3 or sample_seismic_data.shape[0] < config['NUM_SOURCES_TO_ENSEMBLE']:
                            continue

                        source_predictions = []
                        for source_idx in range(config['NUM_SOURCES_TO_ENSEMBLE']):
                            shot_gather = torch.from_numpy(sample_seismic_data[source_idx]).float()
                            input_tensor = ps_utils.preprocess_seismic_with_attributes(shot_gather, dt=config['DT'])
                            prediction_norm = generalist_model(input_tensor.unsqueeze(0).to(device))
                            source_predictions.append(prediction_norm)
                        ensembled_prediction_norm = torch.stack(source_predictions).mean(dim=0)

                        classifier_outputs = classifier_model(ensembled_prediction_norm)
                        _, predicted_label = torch.max(classifier_outputs, 1)

                        all_preds.append(predicted_label.item())
                        all_labels.append(true_label_idx)

                except Exception as e:
                    print(f"Error procesando el archivo {seis_path} en la muestra {sample_idx}: {e}")
                    continue
                    
    return all_labels, all_preds

if __name__ == '__main__':
    print(f"Usando dispositivo: {config['DEVICE']}")
    print(f"Cargando modelo generalista: {config['GENERALIST_MODEL_PATH']}")
    generalist_model = model_loader.SimpleUnet(
        encoder_name="timm-efficientnet-b7", in_channels=4, out_classes=1, encoder_weights=None
    )
    state_dict_gen = torch.load(config['GENERALIST_MODEL_PATH'], map_location=config['DEVICE'])
    generalist_model.load_state_dict(state_dict_gen.get('model_state_dict', state_dict_gen))
    print("Modelo generalista cargado.")

    class_names, class_to_idx = get_superfamily_info(config['VALIDATION_DATA_PATH'])
    if class_names is None:
        sys.exit("No se pudo continuar debido a un error en la carga de datos.")
    num_classes = len(class_names)

    print(f"Cargando modelo clasificador: {config['CLASSIFIER_MODEL_PATH']}")
    classifier_model = models.resnet18(weights=None)
    classifier_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    classifier_model.fc = nn.Linear(classifier_model.fc.in_features, num_classes)
    classifier_model.load_state_dict(torch.load(config['CLASSIFIER_MODEL_PATH'], map_location=config['DEVICE']))
    print("Modelo clasificador cargado.")

    print("\n--- Iniciando test de estrés End-to-End (generación + clasificación) ---")
    true_labels, predicted_labels = validate_end_to_end(
        generalist_model, 
        classifier_model, 
        config['VALIDATION_DATA_PATH'], 
        config['DEVICE'], 
        config,
        class_to_idx
    )

    if not true_labels:
        print("\nNo se procesaron muestras. finalizando.")
        sys.exit()

    accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels)) * 100
    
    print("\n" + "="*60)
    print("  RESULTADOS DEL TEST DE ESTRÉS DEL SISTEMA COMPLETO")
    print("  (Entrada: datos sísmicos -> generador -> clasificador)")
    print("="*60)
    print(f"\nTotal de muestras validadas: {len(true_labels)}")
    print(f"Accuracy general: {accuracy:.2f}%\n")

    print("Informe de clasificación por súper-familia:")
    print(classification_report(true_labels, predicted_labels, target_names=class_names, zero_division=0))

    print("\nMatriz de confusión:")
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Etiqueta predicha')
    plt.ylabel('Etiqueta real')
    plt.title('Matriz de confusión del sistema end-to-end')
    plt.tight_layout()
    plt.show()
