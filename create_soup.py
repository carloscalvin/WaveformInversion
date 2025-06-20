import torch
import os
import glob
import model as model_loader

def create_model_soup(checkpoint_paths, device="cpu"):
    """
    Crea un "model soup" promediando los pesos de varios checkpoints.

    Args:
        checkpoint_paths (list): Lista de rutas a los ficheros de checkpoint (.pth).
        device (str): Dispositivo en el que cargar los modelos ('cpu' o 'cuda').

    Returns:
        OrderedDict: El state_dict del modelo promediado.
    """
    if not checkpoint_paths:
        print("Advertencia: No se proporcionaron checkpoints para hacer la 'soup'.")
        return None

    print(f"Creando 'soup' con {len(checkpoint_paths)} modelos...")

    soup_state_dict = torch.load(checkpoint_paths[0], map_location=device)
    if 'model_state_dict' in soup_state_dict:
        soup_state_dict = soup_state_dict['model_state_dict']

    for path in checkpoint_paths[1:]:
        checkpoint = torch.load(path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        for key in soup_state_dict:
            if key in state_dict:
                soup_state_dict[key] += state_dict[key]
            else:
                print(f"Advertencia: La clave '{key}' no se encontró en {path}")

    num_models = len(checkpoint_paths)
    for key in soup_state_dict:
        soup_state_dict[key] = soup_state_dict[key] / float(num_models)

    print("'Model soup' creado exitosamente.")
    return soup_state_dict

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINTS_DIR = './checkpoints'
    CHECKPOINT_PATTERN = os.path.join(CHECKPOINTS_DIR, '*.pth')
    SOUP_MODEL_SAVE_PATH = 'models/efficientnet-b7_soup_model.pth'
    checkpoint_files = glob.glob(CHECKPOINT_PATTERN)

    if not checkpoint_files:
        print(f"No se encontraron checkpoints en '{CHECKPOINTS_DIR}' con el patrón '{CHECKPOINT_PATTERN}'.")
    else:
        souped_state_dict = create_model_soup(checkpoint_files, device=DEVICE)

        if souped_state_dict:
            try:
                verification_model = model_loader.SimpleUnet(
                    encoder_name="timm-efficientnet-b7",
                    in_channels=4,
                    out_classes=1,
                    encoder_weights=None
                )
                verification_model.load_state_dict(souped_state_dict)
                verification_model.to(DEVICE)
                print("El state_dict de la 'soup' se cargó correctamente en la arquitectura del modelo.")
                os.makedirs(os.path.dirname(SOUP_MODEL_SAVE_PATH), exist_ok=True)
                torch.save(souped_state_dict, SOUP_MODEL_SAVE_PATH)
                print(f"Modelo 'soup' guardado en: {SOUP_MODEL_SAVE_PATH}")

            except Exception as e:
                print(f"Error al verificar o guardar el modelo 'soup': {e}")
