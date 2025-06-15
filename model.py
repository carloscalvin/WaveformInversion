import torch.nn as nn
import segmentation_models_pytorch as smp

class SimpleUnet(nn.Module):
    def __init__(self, encoder_name, encoder_weights, in_channels=4, out_classes=1):
        """
        Modelo U-Net modificando la primera capa convolucional del encoder para manejar una entrada
        de alta resolución no cuadrada.

        Args:
            in_channels (int): Número de canales de entrada (4 para nuestros atributos).
            out_classes (int): Número de canales de salida (1 para el mapa de velocidad).
        """
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            activation='sigmoid'
        )

        self.final_adapter = nn.AdaptiveAvgPool2d((70, 70))
        self._adapt_encoder_to_full_resolution()

    def _adapt_encoder_to_full_resolution(self):
        """
        Modifica la primera capa convolucional del encoder para manejar una entrada
        de alta resolución no cuadrada (ej. 1000x70) de forma asimétrica.
        """
        try:
            encoder = self.model.encoder
            new_stride = (5, 1)
            print(f"Adaptando el 'stem' del encoder. Stride original: {encoder.conv_stem.stride}")
            encoder.conv_stem.stride = new_stride
            print(f"Nuevo stride del 'stem': {encoder.conv_stem.stride}")
        except AttributeError:
            print(f"AVISO: No se pudo modificar el 'stem' del encoder automáticamente para el backbone {self.model.encoder.name}.")

    def forward(self, x):
        raw_output = self.model(x)
        final_output = self.final_adapter(raw_output)
        return final_output
