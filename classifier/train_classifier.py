import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from tqdm.auto import tqdm

config = {
    'DATA_PATH': './dataset/',
    'DEVICE': "cuda" if torch.cuda.is_available() else "cpu",
    'NUM_EPOCHS': 10,
    'BATCH_SIZE': 128,
    'LEARNING_RATE': 1e-3,
    'VMIN': 1500.0,
    'VMAX': 4500.0,
    'SAMPLES_PER_FILE': 500,
    'MODEL_SAVE_PATH': './models/best_classifier_model.pth'
}

class VelocityMapDataset(Dataset):
    def __init__(self, root_dir, vmin, vmax, samples_per_file):
        self.root_dir = root_dir
        self.vmin = vmin
        self.vmax = vmax
        self.data_cache = {}
        self.samples = []
        print("Iniciando escaneo de directorios y precarga de datos...")

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"Encontradas {len(self.classes)} clases: {self.classes}")

        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            super_class_path = os.path.join(root_dir, class_name)
            subclass_paths = [f"{class_name}_A", f"{class_name}_B"]
            for subclass_path in subclass_paths:
                class_path = os.path.join(super_class_path, subclass_path)
                for filename in os.listdir(class_path):
                    if filename.endswith('.npy'):
                        file_path = os.path.join(class_path, filename)

                        if file_path not in self.data_cache:
                            loaded_data = np.load(file_path).astype(np.float32)
                            self.data_cache[file_path] = torch.from_numpy(loaded_data)
                        for i in range(samples_per_file):
                            self.samples.append((file_path, i, class_idx))

        print(f"\n¡Precarga completa! {len(self.data_cache)} ficheros cargados en RAM.")
        print(f"Tamaño total del dataset: {len(self.samples)} muestras individuales.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, sample_idx_in_file, label = self.samples[idx]
        velocity_maps_batch = self.data_cache[file_path]
        velocity_map = velocity_maps_batch[sample_idx_in_file]
        image_tensor_2d = (velocity_map.squeeze() - self.vmin) / (self.vmax - self.vmin)
        image_tensor_3d = image_tensor_2d.unsqueeze(0)
        return image_tensor_3d, label

print(f"Cargando datos desde: {config['DATA_PATH']}")
full_dataset = VelocityMapDataset(
    root_dir=config['DATA_PATH'], 
    vmin=config['VMIN'], 
    vmax=config['VMAX'],
    samples_per_file=config['SAMPLES_PER_FILE']
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Datos listos. muestras de entrenamiento: {len(train_dataset)}, muestras de validación: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=0)

print("Creando modelo clasificador (ResNet18)...")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_classes = len(full_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.to(config['DEVICE'])
print("Modelo listo.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config['LEARNING_RATE'])
best_accuracy = 0.0

print("\n--- Iniciando entrenamiento del clasificador ---")

for epoch in range(config['NUM_EPOCHS']):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Época {epoch+1}/{config['NUM_EPOCHS']} - Entrenamiento"):
        inputs, labels = inputs.to(config['DEVICE']), labels.to(config['DEVICE'])
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Época {epoch+1}/{config['NUM_EPOCHS']} - Validación"):
            inputs, labels = inputs.to(config['DEVICE']), labels.to(config['DEVICE'])
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_acc = 100 * correct / total
    print(f"Época {epoch+1} completada. Loss: {epoch_loss:.4f} | Accuracy de validación: {epoch_acc:.2f}%")

    if epoch_acc > best_accuracy:
        print(f"¡Mejora en accuracy! {best_accuracy:.2f}% -> {epoch_acc:.2f}%. guardando modelo...")
        best_accuracy = epoch_acc
        os.makedirs(os.path.dirname(config['MODEL_SAVE_PATH']), exist_ok=True)
        torch.save(model.state_dict(), config['MODEL_SAVE_PATH'])

print("\n--- Entrenamiento del clasificador completado ---")
print(f"Mejor accuracy de validación alcanzada: {best_accuracy:.2f}%")
print(f"Mejor modelo guardado en: {config['MODEL_SAVE_PATH']}")
