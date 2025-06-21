# Predicción de mapas de velocidad sísmica

Este proyecto utiliza técnicas de Deep Learning, específicamente una arquitectura U-Net, para abordar el problema de la Inversión de Onda Completa (FWI). El objetivo es predecir un mapa de velocidad del subsuelo 2D a partir de un sismograma 2D (shot gather), tratando el problema como una tarea de traducción de imagen a imagen.

El modelo se desarrolla y entrena utilizando datos del dataset público **OpenFWI**, con el fin de crear una solución rápida y generalizable que pueda servir como una herramienta eficaz para el análisis geofísico preliminar.

## Características clave

* **Ingeniería de características guiada por la física:** En lugar de usar datos sísmicos crudos, se realiza un preprocesamiento para generar 4 canales de entrada (sismograma, envolvente, fase y frecuencia instantánea) utilizando la librería `bruges`, lo que enriquece la información de entrada para el modelo.
* **Arquitectura U-Net moderna:** Se utiliza una U-Net con un backbone preentrenado, implementada a través de la librería `segmentation-models-pytorch` para un desarrollo rápido y robusto.
* **Validación robusta:** La métrica de validación se calcula promediando las predicciones de múltiples fuentes sísmicas (ensembling de fuentes), lo que proporciona una evaluación más estable y fiable del rendimiento del modelo.

## Stack tecnológico

* Python 3.9+
* PyTorch
* NumPy
* Matplotlib
* segmentation-models-pytorch
* bruges
* tqdm
* scikit-image (para métricas como SSIM)

## Estructura del proyecto

```
.
├── dataset/                  # Para datos preprocesados
├── models/                # Para guardar los checkpoints de los modelos entrenados
├── kaggle_submission.py   # Script para generar entregas de Kaggle
├── model.py               # Definición de la arquitectura del modelo (U-Net)
├── ml_utils.py            # Utilidades de ML (Dataset, AugmentationWrapper, etc.)
├── ps_utils.py            # Utilidades de preprocesamiento sísmico (con bruges)
├── preprocess_data.py     # Script para preprocesar datos a ficheros individuales
├── train.py               # Script principal para entrenar el modelo
├── inference.py           # Script para hacer inferencias con un modelo entrenado
├── visualize.py           # Script para visualización y análisis exploratorio de datos (EDA)
├── utils.py               # Funciones de ayuda generales (ploteo, formato, etc.)
└── requirements.txt       # Dependencias del proyecto
```

## Instalación

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/carloscalvin/WaveformInversion.git
cd WaveformInversion
```

### Paso 2: Crear y activar un entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### Paso 3: Instalar las dependencias

```bash
pip install -r requirements.txt
```

## Uso

### Entrenamiento

Para entrenar el modelo, ejecuta el script `train.py`. Asegúrate de que las rutas a los datos estén configuradas correctamente dentro del script.

```bash
python train.py
```

### Inferencia

Para realizar una predicción sobre un nuevo sismograma usando un modelo ya entrenado:

```bash
python inference.py --model_path models/best_model.pth --input_file ruta/a/tu/sismograma.npy
```

## Hoja de Ruta del Proyecto

A continuación se detallan las tareas completadas y los siguientes pasos planeados.

### Realizado

- [x] **1. Análisis exploratorio de datos (EDA)**
- [x] **2. Ingeniería de características con atributos sísmicos**
- [x] **3. Creación del pipeline de entrenamiento (prueba de concepto)**
- [x] **4. Primera iteración y submission a Kaggle**
- [x] **5. Mejora de la robustez del modelo y validación**
- [x] **6. (BLOQUEANTE) Optimizar pipeline de datos: preprocesar a ficheros individuales**

### Pendiente

- [ ] **7. Entrenamiento a escala: obtener baseline con datos completos**
- [ ] **8. Experimentación: probar arquitecturas de modelo más grandes**
- [ ] **9. Experimentación: entrenar con función de pérdida híbrida**
- [ ] **10. Desarrollo del refinador (Test-Time Optimization - TTO)**
- [ ] **11. Evaluación final y submission**

### Mejoras avanzadas (backlog)

- [ ] **12. Experimentación: crear un ensemble de modelos**
- [ ] **13. Análisis: investigar patrones de error del modelo**
- [ ] **14. Optimización: afinar hiperparámetros del modelo base**

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el fichero `LICENSE` para más detalles.
