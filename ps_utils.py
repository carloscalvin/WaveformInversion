import torch
import numpy as np
from bruges.attribute import envelope, instantaneous_phase, instantaneous_frequency
import warnings

def generate_simple_2layer_model(seis_batch, shape=(70, 70), dt=0.001, dx=10):
    """
    Genera un mapa de velocidad simple de 2 capas basado en la velocidad
    de la primera llegada de la onda en un lote de datos sísmicos.

    Args:
        seis_batch (torch.Tensor): Lote de datos sísmicos (B, S, T, R).
        shape (tuple): Dimensiones (altura, anchura) del mapa de velocidad a generar.
        dt (float): Intervalo de tiempo de muestreo en segundos.
        dx (float): Espaciado entre receptores en metros.
    
    Returns:
        torch.Tensor: Un lote de mapas de velocidad predichos.
    """
    batch_size = seis_batch.shape[0]
    num_receivers = seis_batch.shape[3]
    predicted_vel_maps = []

    for i in range(batch_size):
        sample = seis_batch[i]
        # Usamos la primera fuente para la estimación
        first_source_gather = sample[0]

        # Estimación simple de la primera llegada
        # Buscamos el primer instante de tiempo donde la energía supera un umbral
        threshold = torch.max(torch.abs(first_source_gather)) * 0.1
        arrival_times = torch.argmax((torch.abs(first_source_gather) > threshold).float(), dim=0)

        # Usamos el receptor más lejano para una mejor estimación de la pendiente
        furthest_receiver_idx = num_receivers - 1
        time_at_furthest = arrival_times[furthest_receiver_idx].item() * dt
        dist_at_furthest = furthest_receiver_idx * dx
        
        # v = d / t. Si el tiempo es 0, usamos una velocidad por defecto.
        v_top_layer = dist_at_furthest / time_at_furthest if time_at_furthest > 0 else 1500.0
        # Forzamos que la velocidad esté en un rango plausible
        v_top_layer = np.clip(v_top_layer, 1500, 2500)

        # Crear el mapa de 2 capas
        # Ponemos la interfaz a 1/3 de la profundidad
        interface_depth = shape[0] // 3
        vel_map = torch.ones(shape) * 3000.0 # Velocidad por defecto para la capa inferior
        vel_map[:interface_depth, :] = v_top_layer

        predicted_vel_maps.append(vel_map)

    return torch.stack(predicted_vel_maps).unsqueeze(1) # Añadir dim de canal

def preprocess_seismic_with_attributes(shot_gather_tensor, dt=0.001):
    """
    Toma un sismograma 2D y calcula varios atributos sísmicos,
    devolviendo un tensor multi-canal.
    """
    shot_gather_np = shot_gather_tensor.cpu().numpy()
    target_shape = shot_gather_np.shape

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Calcular atributos sísmicos
        env = np.nan_to_num(envelope(shot_gather_np))
        phase = np.nan_to_num(instantaneous_phase(shot_gather_np))
        freq = np.nan_to_num(instantaneous_frequency(shot_gather_np, dt=dt)) 

    # Verificamos si la forma de la frecuencia es diferente y la corregimos si es necesario
    if freq.shape != target_shape:
        # Esto es común para la frecuencia. Rellenamos la última fila para igualar.
        padding_needed = target_shape[0] - freq.shape[0]
        # 'edge' duplica la última fila para rellenar
        freq = np.pad(freq, ((0, padding_needed), (0, 0)), 'edge')

    # Normalizar cada atributo al rango [0, 1]
    def normalize(arr):
        min_val, max_val = arr.min(), arr.max()
        return (arr - min_val) / (max_val - min_val) if max_val > min_val else arr

    raw_norm = normalize(shot_gather_np)
    env_norm = normalize(env)
    phase_norm = normalize(phase)
    freq_norm = normalize(freq)

    # Apilar los atributos como canales
    processed_channels = np.stack([
        raw_norm, env_norm, phase_norm, freq_norm
    ], axis=0)
    
    return torch.from_numpy(processed_channels).float()
