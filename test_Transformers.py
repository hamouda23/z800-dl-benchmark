
from transformers import AutoModel, AutoTokenizer
import torch
import time

print("="*50)
print("TEST BERT - NLP")
print("="*50)

# Charger BERT-base
print("\nChargement BERT-base...")
model = AutoModel.from_pretrained("bert-base-uncased").cuda()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Test inférence
texts = ["Deep learning on HP Z800 workstation"] * 16
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
inputs = {k: v.cuda() for k, v in inputs.items()}

# Warm-up
with torch.no_grad():
    _ = model(**inputs)

# Benchmark
start = time.time()
for _ in range(50):
    with torch.no_grad():
        outputs = model(**inputs)
torch.cuda.synchronize()
bert_time = time.time() - start

print(f"BERT-base inférence:")
print(f"50 itérations: {bert_time:.4f}s")
print(f"Throughput: {50/bert_time:.2f} it/s")
print(f"Latence: {bert_time/50*1000:.2f} ms/batch")
print(f"VRAM utilisée: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
