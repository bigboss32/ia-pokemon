from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import torch
import os
import gc
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


class MetricsTracker:
    def __init__(self):
        self.training_logs = []
        self.start_time = None
        self.end_time = None
        
    def log_metrics(self, logs):
        """Registra las métricas durante el entrenamiento"""
        if 'train_loss' in logs or 'eval_loss' in logs:
            log_entry = {
                'step': logs.get('step', 0),
                'epoch': logs.get('epoch', 0),
                'learning_rate': logs.get('learning_rate', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Agregar métricas disponibles
            if 'train_loss' in logs:
                log_entry['train_loss'] = logs['train_loss']
            if 'eval_loss' in logs:
                log_entry['eval_loss'] = logs['eval_loss']
                
            self.training_logs.append(log_entry)
    
    def save_metrics(self, output_dir):
        """Guarda las métricas en archivos JSON y CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar como JSON
        with open(f"{output_dir}/training_metrics.json", 'w') as f:
            json.dump(self.training_logs, f, indent=2)
        
        # Guardar como CSV
        if self.training_logs:
            df = pd.DataFrame(self.training_logs)
            df.to_csv(f"{output_dir}/training_metrics.csv", index=False)
            print(f"📊 Métricas guardadas en {output_dir}/training_metrics.csv")
    
    def plot_metrics(self, output_dir):
        """Genera gráficos de las métricas"""
        if not self.training_logs:
            print("⚠️ No hay métricas para graficar")
            return
            
        df = pd.DataFrame(self.training_logs)
        
        # Determinar qué métricas están disponibles
        has_train_loss = 'train_loss' in df.columns and df['train_loss'].notna().any()
        has_eval_loss = 'eval_loss' in df.columns and df['eval_loss'].notna().any()
        
        # Configurar subplots según métricas disponibles
        fig_rows = 2 if has_eval_loss else 2
        fig_cols = 2
        
        plt.figure(figsize=(15, 10))
        
        plot_idx = 1
        
        # Gráfico de pérdida de entrenamiento
        if has_train_loss:
            plt.subplot(fig_rows, fig_cols, plot_idx)
            plt.plot(df['step'], df['train_loss'], 'b-', linewidth=2, label='Train Loss')
            if has_eval_loss:
                eval_data = df[df['eval_loss'].notna()]
                if len(eval_data) > 0:
                    plt.plot(eval_data['step'], eval_data['eval_loss'], 'r-', linewidth=2, label='Eval Loss')
            plt.title('Pérdida de Entrenamiento')
            plt.xlabel('Paso')
            plt.ylabel('Pérdida')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Gráfico por épocas
        if has_train_loss:
            plt.subplot(fig_rows, fig_cols, plot_idx)
            plt.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Train Loss')
            if has_eval_loss:
                eval_data = df[df['eval_loss'].notna()]
                if len(eval_data) > 0:
                    plt.plot(eval_data['epoch'], eval_data['eval_loss'], 'r-', linewidth=2, label='Eval Loss')
            plt.title('Pérdida por Época')
            plt.xlabel('Época')
            plt.ylabel('Pérdida')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Gráfico de learning rate
        if 'learning_rate' in df.columns:
            plt.subplot(fig_rows, fig_cols, plot_idx)
            plt.plot(df['step'], df['learning_rate'], 'g-', linewidth=2)
            plt.title('Tasa de Aprendizaje')
            plt.xlabel('Paso')
            plt.ylabel('Learning Rate')
            plt.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Gráfico logarítmico
        if has_train_loss:
            plt.subplot(fig_rows, fig_cols, plot_idx)
            plt.plot(df['step'], df['train_loss'], 'b-', linewidth=2, label='Train Loss')
            if has_eval_loss:
                eval_data = df[df['eval_loss'].notna()]
                if len(eval_data) > 0:
                    plt.plot(eval_data['step'], eval_data['eval_loss'], 'r-', linewidth=2, label='Eval Loss')
            plt.title('Progreso del Entrenamiento (Log Scale)')
            plt.xlabel('Paso')
            plt.ylabel('Pérdida')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/training_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Gráficos guardados en {output_dir}/training_metrics.png")
    
    def print_summary(self):
        """Imprime un resumen de las métricas"""
        if not self.training_logs:
            print("⚠️ No hay métricas para mostrar")
            return
            
        df = pd.DataFrame(self.training_logs)
        
        print("\n" + "="*50)
        print("📊 RESUMEN DE MÉTRICAS DE ENTRENAMIENTO")
        print("="*50)
        
        # Métricas de entrenamiento
        if 'train_loss' in df.columns and df['train_loss'].notna().any():
            train_data = df[df['train_loss'].notna()]
            initial_train_loss = train_data['train_loss'].iloc[0]
            final_train_loss = train_data['train_loss'].iloc[-1]
            min_train_loss = train_data['train_loss'].min()
            max_train_loss = train_data['train_loss'].max()
            
            print(f"📈 ENTRENAMIENTO:")
            print(f"  Pérdida inicial: {initial_train_loss:.4f}")
            print(f"  Pérdida final: {final_train_loss:.4f}")
            print(f"  Pérdida mínima: {min_train_loss:.4f}")
            print(f"  Pérdida máxima: {max_train_loss:.4f}")
            print(f"  Mejora total: {((initial_train_loss - final_train_loss) / initial_train_loss * 100):.2f}%")
        
        # Métricas de validación
        if 'eval_loss' in df.columns and df['eval_loss'].notna().any():
            eval_data = df[df['eval_loss'].notna()]
            initial_eval_loss = eval_data['eval_loss'].iloc[0]
            final_eval_loss = eval_data['eval_loss'].iloc[-1]
            min_eval_loss = eval_data['eval_loss'].min()
            max_eval_loss = eval_data['eval_loss'].max()
            
            print(f"📊 VALIDACIÓN:")
            print(f"  Pérdida inicial: {initial_eval_loss:.4f}")
            print(f"  Pérdida final: {final_eval_loss:.4f}")
            print(f"  Pérdida mínima: {min_eval_loss:.4f}")
            print(f"  Pérdida máxima: {max_eval_loss:.4f}")
            print(f"  Mejora total: {((initial_eval_loss - final_eval_loss) / initial_eval_loss * 100):.2f}%")
        
        print(f"📊 GENERAL:")
        print(f"  Total de pasos: {len(df)}")
        
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            print(f"  Duración del entrenamiento: {duration:.2f} segundos")
        print("="*50)


class CustomTrainer(Trainer):
    def __init__(self, metrics_tracker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_tracker = metrics_tracker
        
    def log(self, logs):
        super().log(logs)
        self.metrics_tracker.log_metrics(logs)


# Configuración inicial
model_name = "gpt2-xl"
output_dir = "./pokemon_gpt2xl"
print(f"🚀 Usando modelo: {model_name}")

# Configurar CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# Inicializar tracker de métricas
metrics_tracker = MetricsTracker()

# Cargar tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ tokenizer.pad_token_id: {tokenizer.pad_token_id}")

# Cargar modelo
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

model.config.pad_token_id = tokenizer.pad_token_id
# FIX: Comentar use_cache para evitar conflictos con gradient checkpointing
# model.config.use_cache = True

print(f"model.config.pad_token_id: {model.config.pad_token_id}")
print(f"🖥️ Usando dispositivo: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"💾 VRAM total: {total_memory:.1f} GB")

# Configurar LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj", "c_fc"]
)

model = get_peft_model(model, peft_config)

# FIX: Asegurar que los parámetros LoRA requieren gradientes
for param in model.parameters():
    if param.requires_grad:
        param.requires_grad = True

model.train()
model.print_trainable_parameters()

# Verificar que hay parámetros entrenables
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"🔧 Parámetros entrenables: {trainable_params:,}")
if trainable_params == 0:
    print("❌ ERROR: No hay parámetros entrenables!")
    exit(1)

def read_qa_txt(file_path):
    """Lee archivo con formato:
    ###
    Pregunta: ...
    Respuesta: ...
    ###
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = content.split('###')
    qa_pairs = []
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        lines = block.split('\n')
        question = None
        answer = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Pregunta:'):
                question = line.replace('Pregunta:', '').strip()
            elif line.startswith('Respuesta:'):
                answer = line.replace('Respuesta:', '').strip()
        
        if question and answer:
            formatted_text = f"Pregunta: {question}\nRespuesta: {answer}"
            qa_pairs.append({'text': formatted_text})
    
    return qa_pairs

# Cargar y procesar dataset
dataset = Dataset.from_list(read_qa_txt("fine-tuning/pokemon_descriptions.txt"))

# Dividir dataset en entrenamiento y validación
if len(dataset) > 20:  # Solo dividir si hay suficientes ejemplos
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    print(f"📊 Dataset dividido - Entrenamiento: {len(train_dataset)}, Validación: {len(eval_dataset)}")
else:
    train_dataset = dataset
    eval_dataset = None
    print(f"📊 Dataset pequeño ({len(dataset)} ejemplos) - Sin división")

# FIX: Función de tokenización mejorada
def tokenize(examples):
    """Tokeniza los ejemplos asegurando consistencia en la estructura"""
    # Tokenizar con padding consistente
    tokenized = tokenizer(
        examples['text'],
        padding=True,  # Cambiado de False a True
        truncation=True,
        max_length=256,
        return_tensors=None,
        add_special_tokens=True
    )
    
    # Crear labels como copia de input_ids
    tokenized['labels'] = []
    for input_ids in tokenized['input_ids']:
        tokenized['labels'].append(input_ids.copy())
    
    return tokenized

# Aplicar tokenización
train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['text'])
if eval_dataset is not None:
    eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=['text'])

# FIX: Data collator mejorado
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
    return_tensors="pt"  # Asegurar que devuelve tensores PyTorch
)

# Configurar argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,  
    weight_decay=0.01,
    fp16=False,      
    save_strategy="epoch",
    evaluation_strategy="epoch" if eval_dataset is not None else "no",
    logging_steps=5,
    warmup_steps=20,
    lr_scheduler_type="cosine",
    report_to=None,
    gradient_checkpointing=False,  # Cambiado a False para evitar problemas con LoRA
    dataloader_num_workers=0,
    optim="adamw_torch",
    remove_unused_columns=False,  # Cambiado a False
    save_total_limit=2,
    load_best_model_at_end=True if eval_dataset is not None else False,
    metric_for_best_model="eval_loss" if eval_dataset is not None else "train_loss",
    greater_is_better=False,
    dataloader_pin_memory=False,  # Añadido para evitar problemas de memoria
)

def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"💾 VRAM: {allocated:.1f} GB usada, {reserved:.1f} GB reservada")

print_memory_usage()

# Verificar estructura del dataset tokenizado
if len(train_dataset) > 0:
    example = train_dataset[0]
    print(f"📊 Ejemplo tokenizado:")
    print(f"  input_ids type: {type(example['input_ids'])}, length: {len(example['input_ids'])}")
    print(f"  labels type: {type(example['labels'])}, length: {len(example['labels'])}")
    print(f"  attention_mask type: {type(example['attention_mask'])}, length: {len(example['attention_mask'])}")
    
    # Verificar que todos los ejemplos tienen la misma longitud
    lengths = [len(ex['input_ids']) for ex in train_dataset]
    print(f"  Longitudes únicas: {set(lengths)}")
    
    # Mostrar contenido del primer ejemplo
    print(f"  Texto: {tokenizer.decode(example['input_ids'][:50])}...")

# Crear trainer personalizado
trainer = CustomTrainer(
    metrics_tracker=metrics_tracker,
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print(f"📊 Dataset - Entrenamiento: {len(train_dataset)} ejemplos")
if eval_dataset is not None:
    print(f"📊 Dataset - Validación: {len(eval_dataset)} ejemplos")

# Entrenar el modelo
try:
    print("🔥 Entrenando...")
    metrics_tracker.start_time = datetime.now()
    trainer.train()
    metrics_tracker.end_time = datetime.now()

    # Guardar modelo
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Guardar métricas
    metrics_tracker.save_metrics(output_dir)
    metrics_tracker.plot_metrics(output_dir)
    metrics_tracker.print_summary()
    
    print("✅ Entrenamiento completado!")

except torch.cuda.OutOfMemoryError as e:
    print(f"❌ Out of Memory: {e}")
    print("💡 Prueba reducir batch_size o usar gradient_checkpointing=True")
    print("💡 O reduce per_device_train_batch_size a 1 y aumenta gradient_accumulation_steps")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Prueba del modelo
try:
    model.eval()
    test_prompt = "Pregunta: Descríbeme a Pikachu\nRespuesta:"
    inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
    
    # Mover inputs a device
    if torch.cuda.is_available():
        inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"Input IDs: {inputs['input_ids']}")
    print(f"Shape: {inputs['input_ids'].shape}")
    assert inputs['input_ids'].shape[1] > 0, "Prompt vacío!"

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✨ Resultado: {generated_text}")

except Exception as e:
    print(f"❌ Error en prueba: {e}")
    import traceback
    traceback.print_exc()

# Limpieza final
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print_memory_usage()

print("🎉 Script terminado!")