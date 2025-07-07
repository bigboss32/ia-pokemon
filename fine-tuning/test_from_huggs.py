from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


base_model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
model = PeftModel.from_pretrained(base_model, "gelan32/pokemon-gp2")
prompt = "Pregunta: Describe a Pikachu\nRespuesta:"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=50
)


print(tokenizer.decode(outputs[0], skip_special_tokens=True))




 