import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Cargar modelo y tokenizador
modelo = "Llama3.2:1b"  # Puedes cambiarlo por otro LLM
tokenizer = AutoTokenizer.from_pretrained(modelo)
model = AutoModelForCausalLM.from_pretrained(modelo)



# Calcular métricas

num_tokens = len(tokenizer.convert_ids_to_tokens("hola como estas"))

print(f"Latencia: {4:.4f} segundos")
print(f"Número de tokens generados: {num_tokens}")
