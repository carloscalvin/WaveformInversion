import numpy as np
import matplotlib.pyplot as plt

original_file_path = "dataset/preprocessed_train/Style_B/Style_B_data1_sample_000.npz"
f16_file_path = "dataset/preprocessed_train_f16/Style_B/Style_B_data1_sample_000.npz"

with np.load(original_file_path) as data:
    seismic_f32 = data['seismic_data'].astype(np.float32)

with np.load(f16_file_path) as data:
    seismic_f16 = data['seismic_data']

print("--- Análisis de Precisión de Datos ---")
print(f"Tipo de dato original: {seismic_f32.dtype}")
print(f"Tipo de dato convertido: {seismic_f16.dtype}")

print("\nEstadísticas del Array Original (float32):")
print(f"  - Mín: {np.min(seismic_f32):.4f}, Máx: {np.max(seismic_f32):.4f}, Media: {np.mean(seismic_f32):.4f}")

print("\nEstadísticas del Array Convertido (float16):")
print(f"  - Mín: {np.min(seismic_f16):.4f}, Máx: {np.max(seismic_f16):.4f}, Media: {np.mean(seismic_f16):.4f}")

conversion_error = np.mean(np.abs(seismic_f32 - seismic_f16.astype(np.float32)))
print(f"\nError Absoluto Medio (MAE) introducido por la conversión: {conversion_error:.6f}")

shot_idx = 0 
shot_f32 = seismic_f32[shot_idx]
shot_f16 = seismic_f16[shot_idx].astype(np.float32)
difference = shot_f32 - shot_f16

vmax = np.percentile(shot_f32, 99)

fig, axes = plt.subplots(1, 3, figsize=(24, 7))
axes[0].imshow(shot_f32, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
axes[0].set_title('Original (float32)')

axes[1].imshow(shot_f16, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
axes[1].set_title('Convertido (float16)')

im = axes[2].imshow(difference, cmap='seismic', aspect='auto')
axes[2].set_title('Diferencia (Error de Conversión)')
fig.colorbar(im, ax=axes[2])

plt.show()