# 🎮 Fine-tuning GPT-2 XL para Descripciones de Pokémon

Este proyecto implementa un fine-tuning del modelo GPT-2 XL utilizando LoRA (Low-Rank Adaptation) para generar descripciones de Pokémon. El modelo se entrena con un dataset de preguntas y respuestas sobre Pokémon.

## 🚀 Características

- **Modelo base**: GPT-2 XL (1.5B parámetros)
- **Técnica**: LoRA fine-tuning para eficiencia de memoria
- **Métricas**: Seguimiento completo del entrenamiento con gráficos
- **Optimización**: Configuraciones para GPU con memoria limitada
- **Evaluación**: Pruebas automáticas post-entrenamiento

## 🏗️ Arquitectura del Sistema

### Arquitectura General
```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE DE FINE-TUNING                  │
├─────────────────────────────────────────────────────────────┤
│  📁 Dataset                                                 │
│  └── pokemon_descriptions.txt                              │
│       ├── Pregunta: ¿Qué es Pikachu?                      │
│       └── Respuesta: Pikachu es un Pokémon eléctrico...   │
│                           │                                 │
│                           ▼                                 │
│  🔄 Procesamiento                                           │
│  └── Tokenización con AutoTokenizer                        │
│       ├── Padding y truncamiento                           │
│       ├── Conversión a input_ids                           │
│       └── Generación de labels                             │
│                           │                                 │
│                           ▼                                 │
│  🧠 Modelo GPT-2 XL + LoRA                                 │
│  └── Arquitectura Transformer                              │
│       ├── 48 capas de atención                             │
│       ├── 1600 dimensiones ocultas                         │
│       ├── 25 cabezas de atención                           │
│       └── Adaptadores LoRA en capas específicas            │
│                           │                                 │
│                           ▼                                 │
│  📊 Métricas y Monitoreo                                   │
│  └── MetricsTracker                                        │
│       ├── Pérdida de entrenamiento                         │
│       ├── Learning rate                                    │
│       ├── Gráficos automáticos                            │
│       └── Resumen de rendimiento                           │
│                           │                                 │
│                           ▼                                 │
│  💾 Modelo Entrenado                                       │
│  └── Adaptadores LoRA + Tokenizer                          │
│       ├── adapter_model.bin                                │
│       ├── adapter_config.json                              │
│       └── Métricas de entrenamiento                        │
└─────────────────────────────────────────────────────────────┘
```

### Arquitectura GPT-2 XL
```
┌─────────────────────────────────────────────────────────────┐
│                      GPT-2 XL ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────┤
│  📊 Especificaciones                                        │
│  ├── Parámetros: 1.5B                                      │
│  ├── Capas: 48                                             │
│  ├── Dimensiones ocultas: 1600                             │
│  ├── Cabezas de atención: 25                               │
│  ├── Vocabulario: 50,257 tokens                            │
│  └── Contexto máximo: 1024 tokens                          │
│                                                             │
│  🔄 Flujo de datos                                          │
│  Input Text → Tokenizer → Embeddings → Transformer Blocks  │
│                                                             │
│  📝 Bloque Transformer (x48)                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ LayerNorm                                           │   │
│  │           ↓                                         │   │
│  │ Multi-Head Attention ← LoRA (c_attn)               │   │
│  │           ↓                                         │   │
│  │ Residual Connection                                 │   │
│  │           ↓                                         │   │
│  │ LayerNorm                                           │   │
│  │           ↓                                         │   │
│  │ Feed Forward (c_fc) ← LoRA                         │   │
│  │           ↓                                         │   │
│  │ Projection (c_proj) ← LoRA                         │   │
│  │           ↓                                         │   │
│  │ Residual Connection                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  🎯 Cabeza de salida                                        │
│  Final LayerNorm → Linear Layer → Softmax → Probabilidades │
└─────────────────────────────────────────────────────────────┘
```

### Arquitectura LoRA (Low-Rank Adaptation)
```
┌─────────────────────────────────────────────────────────────┐
│                       LoRA ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│  🎯 Concepto                                                │
│  En lugar de entrenar toda la matriz W, LoRA entrena       │
│  dos matrices más pequeñas A y B tal que: W' = W + BA      │
│                                                             │
│  📊 Configuración                                           │
│  ├── Rango (r): 16                                         │
│  ├── Alpha: 32                                             │
│  ├── Dropout: 0.1                                          │
│  └── Módulos objetivo: c_attn, c_proj, c_fc                │
│                                                             │
│  🔄 Flujo LoRA                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │  Input (d)                                          │   │
│  │     │                                               │   │
│  │     ├─────────────────────────────────────────────┐ │   │
│  │     │                                             │ │   │
│  │     ▼                                             │ │   │
│  │  Original Weight W                                 │ │   │
│  │  (frozen, no training)                            │ │   │
│  │     │                                             │ │   │
│  │     │                                             │ │   │
│  │     ▼                                             │ │   │
│  │  LoRA Branch A (d × r)                           │ │   │
│  │     │                                             │ │   │
│  │     ▼                                             │ │   │
│  │  LoRA Branch B (r × d)                           │ │   │
│  │     │                                             │ │   │
│  │     ▼                                             │ │   │
│  │  Scale by α/r                                    │ │   │
│  │     │                                             │ │   │
│  │     └─────────────────────────────────────────────┘ │   │
│  │                     │                               │   │
│  │                     ▼                               │   │
│  │                  Add (+)                            │   │
│  │                     │                               │   │
│  │                     ▼                               │   │
│  │                  Output                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  💡 Ventajas                                                │
│  ├── Solo entrena ~1% de los parámetros originales         │
│  ├── Reduce significativamente el uso de memoria           │
│  ├── Permite fine-tuning eficiente                         │
│  └── Mantiene la calidad del modelo base                   │
└─────────────────────────────────────────────────────────────┘
```

### Arquitectura de Entrenamiento
```
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🔄 Flujo de Entrenamiento                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │  Batch de datos                                     │   │
│  │       │                                             │   │
│  │       ▼                                             │   │
│  │  Forward Pass                                       │   │
│  │  ├── Tokenización                                   │   │
│  │  ├── Embeddings                                     │   │
│  │  ├── Transformer Blocks (con LoRA)                  │   │
│  │  └── Predicción de siguiente token                  │   │
│  │       │                                             │   │
│  │       ▼                                             │   │
│  │  Cálculo de pérdida                                 │   │
│  │  └── CrossEntropyLoss                               │   │
│  │       │                                             │   │
│  │       ▼                                             │   │
│  │  Backward Pass                                      │   │
│  │  ├── Gradientes solo para parámetros LoRA           │   │
│  │  └── Parámetros base congelados                     │   │
│  │       │                                             │   │
│  │       ▼                                             │   │
│  │  Actualización de parámetros                       │   │
│  │  └── AdamW Optimizer                                │   │
│  │       │                                             │   │
│  │       ▼                                             │   │
│  │  Métricas y logging                                 │   │
│  │  └── MetricsTracker                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  📊 Componentes del sistema                                 │
│  ├── DataCollatorForLanguageModeling                       │
│  │   └── Maneja padding y creación de batches             │
│  ├── Custom Trainer                                        │
│  │   └── Extiende Trainer con métricas personalizadas     │
│  ├── MetricsTracker                                        │
│  │   └── Registra y visualiza métricas                    │
│  └── Schedulers                                            │
│      └── Cosine annealing para learning rate               │
└─────────────────────────────────────────────────────────────┘
```

### Arquitectura de Datos
```
┌─────────────────────────────────────────────────────────────┐
│                     DATA ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📁 Estructura de datos                                     │
│  pokemon_descriptions.txt                                   │
│  ├── Formato: Bloques separados por ###                    │
│  ├── Pregunta: [texto]                                     │
│  ├── Respuesta: [texto]                                    │
│  └── Encoding: UTF-8                                       │
│                                                             │
│  🔄 Pipeline de procesamiento                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │  Texto crudo                                        │   │
│  │       │                                             │   │
│  │       ▼                                             │   │
│  │  Parsing (read_qa_txt)                             │   │
│  │  ├── Split por ###                                  │   │
│  │  ├── Extracción de pregunta                         │   │
│  │  ├── Extracción de respuesta                        │   │
│  │  └── Formato: "Pregunta: X\nRespuesta: Y"          │   │
│  │       │                                             │   │
│  │       ▼                                             │   │
│  │  Tokenización                                       │   │
│  │  ├── AutoTokenizer de GPT-2                         │   │
│  │  ├── Padding: False (dinámico)                      │   │
│  │  ├── Truncation: True (max_length=256)              │   │
│  │  └── Special tokens: Añadidos                       │   │
│  │       │                                             │   │
│  │       ▼                                             │   │
│  │  Dataset de Hugging Face                            │   │
│  │  ├── input_ids: Tokens de entrada                   │   │
│  │  ├── attention_mask: Máscara de atención            │   │
│  │  └── labels: Tokens objetivo (copia de input_ids)   │   │
│  │       │                                             │   │
│  │       ▼                                             │   │
│  │  DataCollator                                       │   │
│  │  ├── Padding dinámico a múltiplos de 8              │   │
│  │  ├── Creación de batches                            │   │
│  │  └── Preparación para GPU                           │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Requisitos

### Dependencias principales
```bash
pip install transformers
pip install peft
pip install datasets
pip install torch
pip install matplotlib
pip install pandas
```

### Requisitos del sistema
- **GPU**: Recomendado 8GB+ VRAM
- **RAM**: 16GB+ recomendado
- **Almacenamiento**: 10GB+ libres

## 📁 Estructura del proyecto

```
pokemon-finetuning/
├── fine-tuning/
│   └── pokemon_descriptions.txt    # Dataset de entrenamiento
├── pokemon_gpt2xl/                 # Modelo entrenado (se genera)
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── training_metrics.json
│   ├── training_metrics.csv
│   └── training_metrics.png
├── fine_tuning_script.py           # Script principal
└── README.md                       # Este archivo
```

## 🎯 Formato del dataset

El archivo `pokemon_descriptions.txt` debe tener el siguiente formato:

```
###
Pregunta: ¿Qué tipo de Pokémon es Pikachu?
Respuesta: Pikachu es un Pokémon de tipo Eléctrico conocido por sus mejillas que almacenan electricidad.
###

###
Pregunta: Describe a Charizard
Respuesta: Charizard es un Pokémon de tipo Fuego/Volador con forma de dragón que puede volar y lanzar llamas.
###
```

## 🔧 Configuración

### Parámetros de LoRA
- `r`: 16 (rango de las matrices de bajo rango)
- `lora_alpha`: 32 (factor de escalamiento)
- `lora_dropout`: 0.1 (dropout para regularización)
- `target_modules`: ["c_attn", "c_proj", "c_fc"]

### Hiperparámetros de entrenamiento
- `num_train_epochs`: 5
- `per_device_train_batch_size`: 1
- `gradient_accumulation_steps`: 4
- `learning_rate`: 5e-5
- `weight_decay`: 0.01
- `lr_scheduler_type`: "cosine"

## 🚀 Uso

### 1. Preparar el dataset
Asegúrate de tener el archivo `fine-tuning/pokemon_descriptions.txt` con el formato correcto.

### 2. Ejecutar el entrenamiento
```bash
python fine_tuning_script.py
```

### 3. Monitorear el progreso
El script mostrará:
- Uso de memoria GPU
- Pérdida de entrenamiento en tiempo real
- Progreso por épocas

### 4. Resultados
Después del entrenamiento encontrarás:
- Modelo entrenado en `./pokemon_gpt2xl/`
- Métricas en `training_metrics.json` y `training_metrics.csv`
- Gráficos en `training_metrics.png`

## 📊 Métricas de entrenamiento

El script genera automáticamente:

### Gráficos
- **Pérdida por paso**: Evolución de la pérdida durante el entrenamiento
- **Pérdida por época**: Progreso general del modelo
- **Tasa de aprendizaje**: Evolución del learning rate
- **Progreso (escala log)**: Vista logarítmica del progreso

### Archivos de métricas
- **JSON**: Datos completos con timestamps
- **CSV**: Datos tabulares para análisis posterior

### Resumen automático
```
📊 RESUMEN DE MÉTRICAS DE ENTRENAMIENTO
==================================================
Pérdida inicial: 3.2450
Pérdida final: 1.8920
Pérdida mínima: 1.8920
Pérdida máxima: 3.2450
Mejora total: 41.72%
Total de pasos: 125
Duración del entrenamiento: 1847.32 segundos
==================================================
```

## 🧪 Evaluación

El script incluye una prueba automática que:
1. Carga el modelo entrenado
2. Genera una respuesta a una pregunta de prueba
3. Muestra el resultado generado

Ejemplo de salida:
```
✨ Resultado: Pregunta: Descríbeme a Pikachu
Respuesta: Pikachu es un Pokémon de tipo Eléctrico pequeño y amarillo...
```

## ⚙️ Optimizaciones para GPU limitada

### Si tienes problemas de memoria:
1. **Reduce batch_size**: Cambia `per_device_train_batch_size` a 1
2. **Aumenta gradient_accumulation**: Incrementa `gradient_accumulation_steps`
3. **Habilita gradient checkpointing**: `gradient_checkpointing=True`
4. **Usa FP16**: `fp16=True` (en algunas configuraciones)

### Variables de entorno útiles:
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

## 🔍 Troubleshooting

### Error: CUDA Out of Memory
```bash
# Soluciones:
1. Reduce batch_size a 1
2. Aumenta gradient_accumulation_steps
3. Habilita gradient_checkpointing
4. Cierra otros procesos que usen GPU
```

### Error: Dataset vacío
```bash
# Verifica:
1. El archivo pokemon_descriptions.txt existe
2. El formato es correcto (###, Pregunta:, Respuesta:)
3. El encoding es UTF-8
```

### Error: Tokenizer pad_token
```bash
# El script maneja esto automáticamente:
tokenizer.pad_token = tokenizer.eos_token
```

## 📈 Interpretación de resultados

### Métricas importantes:
- **Pérdida decreciente**: Indica que el modelo está aprendiendo
- **Mejora > 30%**: Buen progreso en el entrenamiento
- **Pérdida estable**: El modelo ha convergido

### Señales de alerta:
- **Pérdida aumenta**: Posible overfitting o learning rate alto
- **Pérdida se estanca**: Modelo no está aprendiendo
- **Generación repetitiva**: Ajustar temperature o top_k

## 🤝 Contribuciones


Para mejorar el proyecto:
1. Agrega más datos de entrenamiento
2. Experimenta con hiperparámetros
3. Implementa métricas adicionales (BLEU, ROUGE)
4. Agrega validación cruzada

## 📜 Licencia

Este proyecto está bajo la licencia MIT. Ver LICENSE para más detalles.

## 🙏 Reconocimientos

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PEFT](https://github.com/huggingface/peft)
- [OpenAI GPT-2](https://openai.com/blog/better-language-models/)

---

**Nota**: Este es un proyecto educativo para aprender sobre fine-tuning de modelos de lenguaje. Los resultados pueden variar según el hardware y los datos utilizados.