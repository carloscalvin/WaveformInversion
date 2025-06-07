import torch
import os
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from datetime import datetime

# 1. Recargar todos nuestros módulos
import ml_utils, ps_utils, model, utils

# --- CONFIGURACIÓN DEL ENTRENAMIENTO ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")
PATH_TO_FLATVEL_A = 'kaggle/input/train_samples/FlatVel_A'
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
VMIN, VMAX = 1500.0, 4500.0
VELOCITY_RANGE = VMAX - VMIN
MODELS_DIR = 'models'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, f'best_unet_model_{timestamp}.pth')
print(f"El mejor modelo se guardará en: {MODEL_SAVE_PATH}")

# --- PREPARACIÓN DE DATOS ---
# Instanciamos nuestro dataset solo con la familia FlatVel_A
full_dataset = ml_utils.SeismicDataset(
    data_family_paths=[PATH_TO_FLATVEL_A],
    preprocess_function=ps_utils.preprocess_seismic_with_attributes,
    vmin=VMIN, vmax=VMAX
)

# Dividimos en entrenamiento (80%) y validación (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Creamos los DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

print(f"\nDatos listos. Muestras de entrenamiento: {len(train_dataset)}, Muestras de validación: {len(val_dataset)}")

# --- INICIALIZACIÓN DEL MODELO ---
unet_model = model.SimpleUnet(in_channels=4, out_classes=1).to(DEVICE)
loss_fn = nn.L1Loss() # L1Loss es el MAE, perfecto para nuestra métrica
optimizer = AdamW(unet_model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

print("Modelo, función de pérdida y optimizador inicializados.")

# --- BUCLE DE ENTRENAMIENTO Y VALIDACIÓN ---
train_mae_history = []
val_mae_history = []
best_val_mae = float('inf')

for epoch in range(NUM_EPOCHS):
    print(f"\n--- Iniciando Época {epoch+1}/{NUM_EPOCHS} ---")

    # Fase de Entrenamiento
    unet_model.train()
    train_loss_acum = 0.0
    for inputs, targets in tqdm(train_loader, desc="Entrenamiento"):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        predictions = unet_model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        train_loss_acum += loss.item()

    avg_train_loss = train_loss_acum / len(train_loader)

    # Fase de Validación
    unet_model.eval()
    val_loss_acum = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validación"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            predictions = unet_model(inputs)
            loss = loss_fn(predictions, targets)
            val_loss_acum += loss.item()

    avg_val_mae = val_loss_acum / len(val_loader)

    avg_train_mae_denorm = avg_train_loss * VELOCITY_RANGE
    avg_val_mae_denorm = avg_val_mae * VELOCITY_RANGE
    train_mae_history.append(avg_train_mae_denorm)
    val_mae_history.append(avg_val_mae_denorm)

    print(f"Época {epoch+1} completada. Loss de Entrenamiento: {avg_train_mae_denorm:.2f} | MAE de Validación: {avg_val_mae_denorm:.2f}")

    if avg_val_mae_denorm < best_val_mae:
        print(f"¡Mejora en MAE de validación! {best_val_mae:.2f} -> {avg_val_mae_denorm:.2f}. Guardando modelo...")
        best_val_mae = avg_val_mae_denorm
        torch.save(unet_model.state_dict(), MODEL_SAVE_PATH)
    scheduler.step()

ml_utils.plot_training_history(train_mae_history, val_mae_history)

# --- VISUALIZACIÓN FINAL CON EL MEJOR MODELO ---
print("\n--- Visualizando la predicción del MEJOR modelo en un lote de validación ---")

# 1. Crear una nueva instancia del modelo y cargar los pesos del mejor guardado
best_model = model.SimpleUnet(in_channels=4, out_classes=1)
best_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
best_model.to(DEVICE)
best_model.eval() # Poner en modo evaluación

# 2. Obtener un lote de datos del conjunto de validación
with torch.no_grad():
    inputs_norm, targets_norm = next(iter(val_loader))
    inputs_norm = inputs_norm.to(DEVICE)

    # 3. Hacer una predicción con el mejor modelo
    predictions_norm = best_model(inputs_norm).cpu()

    # 4. Desnormalizar todo para una visualización en la escala correcta (m/s)
    def denormalize(tensor):
        return tensor * VELOCITY_RANGE + VMIN

    targets_denorm = denormalize(targets_norm)
    prediction_denorm = denormalize(predictions_norm)
    
    # 5. Visualizar la comparación para una muestra del lote
    sample_idx_to_show = 5 
    utils.plot_map_comparison(
        map1=targets_denorm[sample_idx_to_show, 0],
        title1=f'Mapa Real (Validación, Muestra #{sample_idx_to_show})',
        map2=prediction_denorm[sample_idx_to_show, 0],
        title2=f'Predicción del Modelo U-Net\n(Mejor MAE: {best_val_mae:.2f} m/s)',
        main_title='Comparación Final: Real vs. Predicción del Mejor Modelo'
    )
