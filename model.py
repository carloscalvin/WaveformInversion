import torch.nn as nn
import segmentation_models_pytorch as smp

class EfficientNet12(nn.Module):
    def __init__(self, in_channels=4, out_classes=1, encoder_weights="noisy-student"):
        """
        Modelo U-Net simple usando segmentation-models-pytorch.
        
        Args:
            in_channels (int): Número de canales de entrada (4 para nuestros atributos).
            out_classes (int): Número de canales de salida (1 para el mapa de velocidad).
        """
        super().__init__()

        self.model = smp.Unet(
            encoder_name="timm-efficientnet-l2",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            activation='sigmoid'
        )

    def forward(self, x):
        return self.model(x)
