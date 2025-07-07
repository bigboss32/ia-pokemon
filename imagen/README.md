# Pokemon Multi-Label Classifier

## 📋 Descripción

Este proyecto implementa un clasificador multi-etiqueta para Pokémon utilizando deep learning. El modelo es capaz de identificar múltiples tipos de Pokémon en una sola imagen (por ejemplo, un Pokémon puede ser tanto "Water" como "Flying").

## 🏗️ Arquitectura del Modelo

### Red Neuronal Base
- **Modelo:** ResNet-18 pre-entrenado en ImageNet
- **Backbone:** Convolutional Neural Network con 18 capas
- **Transfer Learning:** Utiliza pesos pre-entrenados de ImageNet para aprovechar características visuales ya aprendidas

### Modificaciones para Multi-Label
- **Capa de Salida:** Reemplaza la capa fully connected final con una nueva capa linear
- **Dimensiones de Salida:** 18 neuronas (una por cada tipo de Pokémon)
- **Función de Activación:** Sigmoid aplicada durante la inferencia (no en el entrenamiento)
- **Función de Pérdida:** BCEWithLogitsLoss (Binary Cross-Entropy with Logits)

### Tipos de Pokémon Soportados
```
normal, fire, water, electric, grass, ice, fighting, poison, 
ground, flying, psychic, bug, rock, ghost, dragon, dark, steel, fairy
```

## 🔧 Componentes Técnicos

### Preprocesamiento de Datos
```python
transforms.Compose([
    transforms.Resize((224, 224)),          # Redimensiona a entrada estándar
    transforms.RandomHorizontalFlip(),      # Augmentación: flip horizontal
    transforms.RandomRotation(10),          # Augmentación: rotación ±10°
    transforms.ToTensor(),                  # Conversión a tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalización ImageNet
])
```

### Configuración de Entrenamiento
- **Batch Size:** 16
- **Épocas:** 50
- **Optimizador:** Adam con learning rate 1e-5
- **División de Datos:** 80% entrenamiento, 20% validación
- **Fine-tuning:** Completo (todos los parámetros entrenables)

### Umbral de Decisión
- **Threshold:** 0.5 para convertir probabilidades en predicciones binarias
- **Criterio:** `sigmoid(output) > 0.5` determina si un tipo está presente

## 📊 Métricas de Evaluación

El modelo evalúa su rendimiento usando múltiples métricas:

- **Accuracy:** Precisión por imagen (porcentaje de tipos correctamente clasificados)
- **Precision:** Precisión macro-promedio entre todos los tipos
- **Recall:** Recall macro-promedio entre todos los tipos  
- **F1-Score:** Media armónica entre precisión y recall

## 📁 Estructura del Proyecto

```
proyecto/
├── pokemon_images_by_type/     # Dataset organizado por tipos
├── uno.py                      # Clase PokemonMultiLabelDataset
├── main.py                     # Script principal de entrenamiento
├── pokemon_multi_label.pth     # Modelo entrenado (generado)
├── training_metrics.csv        # Métricas por época (generado)
├── loss_curve.png             # Gráfica de pérdida (generado)
├── accuracy_curve.png         # Gráfica de precisión (generado)
└── predictions/               # Ejemplos de predicciones (generado)
```

## 🚀 Uso del Modelo

### Entrenamiento
```bash
python main.py
```

### Salidas Generadas
1. **Modelo entrenado:** `pokemon_multi_label.pth`
2. **Métricas:** `training_metrics.csv` con histórico de entrenamiento
3. **Gráficas:** Curvas de pérdida y precisión
4. **Predicciones:** 5 ejemplos visuales en carpeta `predictions/`

### Inferencia
```python
# Cargar modelo
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 18)
model.load_state_dict(torch.load("pokemon_multi_label.pth"))
model.eval()

# Predecir
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.sigmoid(outputs)
    predictions = probabilities > 0.5
```

## 🎯 Características Clave

- **Multi-Label:** Puede predecir múltiples tipos simultáneamente
- **Data Augmentation:** Mejora la generalización del modelo
- **Transfer Learning:** Acelera el entrenamiento y mejora el rendimiento
- **Monitoreo Completo:** Registra múltiples métricas durante el entrenamiento
- **Visualización:** Genera gráficas y ejemplos de predicciones

## ⚙️ Requisitos

- PyTorch
- torchvision
- scikit-learn
- matplotlib
- numpy

## 📈 Optimizaciones Implementadas

1. **Early Stopping:** Guarda el mejor modelo basado en validation loss
2. **Learning Rate Bajo:** 1e-5 para fine-tuning estable
3. **Normalización:** Usa estadísticas de ImageNet para mejor transferencia
4. **Batch Processing:** Procesamiento eficiente por lotes