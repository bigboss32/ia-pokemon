from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import torch
import os
import gc


model_name = "gpt2-xl"
print(f"🚀 Usando modelo: {model_name}")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()


tokenizer = AutoTokenizer.from_pretrained(model_name)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ tokenizer.pad_token_id: {tokenizer.pad_token_id}")


device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = True

print(f"model.config.pad_token_id: {model.config.pad_token_id}")

print(f" Usando dispositivo: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"💾 VRAM total: {total_memory:.1f} GB")


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj", "c_fc"]
)

model = get_peft_model(model, peft_config)
model.train()

model.print_trainable_parameters()
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

dataset = Dataset.from_list(read_qa_txt("fine-tuning/pokemon_descriptions.txt"))

def tokenize(examples):
    tokenized = tokenizer(
        examples['text'],
        padding=False,
        truncation=True,
        max_length=256,
        return_tensors=None,
        add_special_tokens=True
    )
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

dataset = dataset.map(tokenize, batched=True, remove_columns=['text'])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

training_args = TrainingArguments(
    output_dir="./pokemon_gpt2xl",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,  
    weight_decay=0.01,
    fp16=False,      
    save_strategy="epoch",
    logging_steps=5,
    warmup_steps=20,
    lr_scheduler_type="cosine",
    report_to=None,
    gradient_checkpointing=False,  
    dataloader_num_workers=0,
    optim="adamw_torch",
    remove_unused_columns=True,
)

def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"VRAM: {allocated:.1f} GB usada, {reserved:.1f} GB reservada")

print_memory_usage()

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

print(f"📊 Dataset: {len(dataset)} ejemplos")

if len(dataset) > 0:
    example = dataset[0]
    print(f"input_ids: {len(example['input_ids'])}, labels: {len(example['labels'])}")
    print(f"Texto: {tokenizer.decode(example['input_ids'][:50])}...")

try:
    print(" Entrenando...")
    trainer.train()

    model.save_pretrained("./pokemon_gpt2xl")
    tokenizer.save_pretrained("./pokemon_gpt2xl")
    print("Entrenamiento completado!")

except torch.cuda.OutOfMemoryError as e:
    print(f" Out of Memory: {e}")

except Exception as e:
    print(f" Error: {e}")


try:
    model.eval()
    test_prompt = "Pregunta: Descríbeme a Pikachu\nRespuesta:"
    inputs = tokenizer(test_prompt, return_tensors="pt")
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
    print(f" Error en prueba: {e}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print_memory_usage()

print(" Script terminado!")