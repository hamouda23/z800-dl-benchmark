from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf
import time
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
JSON_FILE = os.path.join(RESULTS_DIR, "test_keras_nlp.json")
TXT_FILE  = os.path.join(RESULTS_DIR, "test_keras_nlp.txt")

print("="*50)
print("TEST BERT - NLP (Keras/TensorFlow)")
print("="*50)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("\nChargement BERT-base...")
model = TFAutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("Modèle chargé.")

texts = ["Deep learning on HP Z800 workstation"] * 16
inputs = tokenizer(texts, return_tensors="tf", padding=True,
                   truncation=True, max_length=128)

_ = model(**inputs, training=False)  # warm-up

start = time.time()
for _ in range(50):
    outputs = model(**inputs, training=False)
bert_time = time.time() - start

throughput = round(50 / bert_time, 2)
latency_ms = round(bert_time / 50 * 1000, 2)

print(f"\nBERT-base inférence :")
print(f"50 itérations: {bert_time:.4f}s")
print(f"Throughput: {throughput} it/s")
print(f"Latence: {latency_ms} ms/batch")

mem = tf.config.experimental.get_memory_info('GPU:0')
vram = round(mem['current'] / 1024**3, 2)
print(f"VRAM utilisée: {vram} GB")

results = {
    "timestamp": timestamp,
    "model": "bert-base-uncased",
    "tensorflow": tf.__version__,
    "iterations": 50,
    "batch_size": 16,
    "throughput_it_per_s": throughput,
    "latency_ms": latency_ms,
    "vram_gb": vram
}

# --- Sauvegarde ---
existing = []
if os.path.exists(JSON_FILE):
    with open(JSON_FILE) as f:
        existing = json.load(f)
existing.append(results)
with open(JSON_FILE, "w") as f:
    json.dump(existing, f, indent=2)

with open(TXT_FILE, "a") as f:
    f.write(f"\n{'='*50}\n")
    f.write(f"Run: {timestamp}\n")
    f.write(f"Modèle     : bert-base-uncased\n")
    f.write(f"Throughput : {throughput} it/s\n")
    f.write(f"Latence    : {latency_ms} ms/batch\n")
    f.write(f"VRAM       : {vram} GB\n")

print(f"\nRésultats sauvegardés dans {RESULTS_DIR}")
