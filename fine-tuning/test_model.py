from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_DIR = "./pokemon_gpt2xl"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32  
print(f"Dispositivo: {device}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch_dtype).to(device)

try:
    model.eval()
    test_prompt = "Pregunta:Descríbeme a Gloom"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"Input IDs: {inputs['input_ids']}")
    print(f"Shape: {inputs['input_ids'].shape}")
    assert inputs['input_ids'].shape[1] > 0, "Prompt vacío!"

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=30,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✨ Resultado: {generated_text}")

except Exception as e:
    print(f" Error en prueba: {e}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✅ VRAM liberada")


