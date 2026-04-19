import tensorflow as tf
import time
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
JSON_FILE = os.path.join(RESULTS_DIR, "test_keras_cv.json")
TXT_FILE  = os.path.join(RESULTS_DIR, "test_keras_cv.txt")

print("="*50)
print("TEST COMPUTER VISION - EfficientNetB0 (Keras)")
print("="*50)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("\nChargement EfficientNetB0...")
model = tf.keras.applications.EfficientNetB0(weights=None, include_top=True, classes=1000)
print("Modèle chargé.")

results = {"timestamp": timestamp, "model": "EfficientNetB0", "tensorflow": tf.__version__}

# Test single image
img = tf.random.uniform([1, 224, 224, 3]) / 255.0
_ = model(img, training=False)  # warm-up

start = time.time()
for _ in range(100):
    _ = model(img, training=False)
elapsed = time.time() - start

fps = round(100 / elapsed, 2)
latency_ms = round(elapsed / 100 * 1000, 2)
print(f"\nEfficientNetB0 inférence (1 image):")
print(f"100 itérations: {elapsed:.4f}s")
print(f"FPS: {fps}")
print(f"Latence: {latency_ms} ms/image")

results["single_image"] = {"fps": fps, "latency_ms": latency_ms, "iterations": 100}

# Test batch 16
print("\n--- Test batch (16 images) ---")
batch_img = tf.random.uniform([16, 224, 224, 3]) / 255.0
_ = model(batch_img, training=False)  # warm-up

start = time.time()
for _ in range(50):
    _ = model(batch_img, training=False)
elapsed_batch = time.time() - start

throughput = round(50 * 16 / elapsed_batch, 0)
print(f"50 itérations (batch=16): {elapsed_batch:.4f}s")
print(f"Throughput: {throughput:.0f} images/s")

mem = tf.config.experimental.get_memory_info('GPU:0')
vram = round(mem['current'] / 1024**3, 2)
print(f"VRAM utilisée: {vram} GB")

results["batch_16"] = {"throughput_images_per_s": throughput, "iterations": 50, "vram_gb": vram}

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
    f.write(f"Modèle     : EfficientNetB0\n")
    f.write(f"FPS (x1)   : {fps}\n")
    f.write(f"Latence    : {latency_ms} ms/image\n")
    f.write(f"Throughput : {throughput:.0f} img/s (batch=16)\n")
    f.write(f"VRAM       : {vram} GB\n")

print(f"\nRésultats sauvegardés dans {RESULTS_DIR}")
