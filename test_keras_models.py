import tensorflow as tf
import time
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
JSON_FILE = os.path.join(RESULTS_DIR, "test_keras_models.json")
TXT_FILE  = os.path.join(RESULTS_DIR, "test_keras_models.txt")

print("="*80)
print("TEST PERFORMANCE IA - Quadro P4000 (Keras/TensorFlow)")
print("="*80)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU: {gpus[0].name if gpus else 'Non détecté'}")
print(f"TensorFlow: {tf.__version__}\n")

results = {"timestamp": timestamp, "tensorflow": tf.__version__, "inference": {}, "training": {}}


def benchmark_inference(model_name, model_fn, batch_sizes=[16, 32, 64, 128]):
    print(f"\n--- INFERENCE {model_name.upper()} ---")
    model = model_fn(weights=None, include_top=True)
    batch_results = []

    for bs in batch_sizes:
        try:
            x = tf.random.normal([bs, 224, 224, 3])
            for _ in range(10):
                _ = model(x, training=False)

            tf.config.experimental.reset_memory_stats('GPU:0')
            start = time.time()
            for _ in range(100):
                _ = model(x, training=False)
            elapsed = time.time() - start

            images_per_sec = round((100 * bs) / elapsed, 0)
            mem = tf.config.experimental.get_memory_info('GPU:0')
            vram = round(mem['peak'] / 1024**3, 2)

            print(f"Batch {bs:3d} → {images_per_sec:6.0f} images/s | VRAM pic {vram} Go")
            batch_results.append({"batch_size": bs, "images_per_sec": images_per_sec, "vram_peak_gb": vram})

        except Exception:
            print(f"Batch {bs:3d} → Trop gros pour la VRAM (8 Go)")
            break

    return batch_results


results["inference"]["ResNet50"] = benchmark_inference(
    "ResNet50", tf.keras.applications.ResNet50, batch_sizes=[8, 16, 32, 64])

# Entraînement
print("\n--- ENTRAÎNEMENT ResNet50 ---")
model = tf.keras.applications.ResNet50(weights=None, include_top=True, classes=1000)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

x = tf.random.normal([32, 224, 224, 3])
y = tf.random.uniform([32], minval=0, maxval=1000, dtype=tf.int32)

@tf.function
def train_step(xb, yb):
    with tf.GradientTape() as tape:
        logits = model(xb, training=True)
        loss = loss_fn(yb, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

train_step(x, y)  # warm-up

tf.config.experimental.reset_memory_stats('GPU:0')
start = time.time()
for i in range(100):
    loss = train_step(x, y)
    if i == 0 or i == 99:
        print(f"Itération {i+1:3d} — loss: {loss:.4f}")

elapsed_train = time.time() - start
images_per_sec_train = round((100 * 32) / elapsed_train, 0)
mem = tf.config.experimental.get_memory_info('GPU:0')
vram_train = round(mem['peak'] / 1024**3, 2)

print(f"\nENTRAÎNEMENT terminé :")
print(f"   {images_per_sec_train:.0f} images/s (batch 32)")
print(f"   VRAM max : {vram_train} Go")

results["training"]["ResNet50"] = {
    "images_per_sec": images_per_sec_train,
    "batch_size": 32,
    "iterations": 100,
    "vram_peak_gb": vram_train
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
    f.write(f"\n{'='*80}\n")
    f.write(f"Run: {timestamp}\n")
    f.write(f"TensorFlow : {tf.__version__}\n")
    f.write(f"\nINFERENCE ResNet50:\n")
    for b in results["inference"]["ResNet50"]:
        f.write(f"  Batch {b['batch_size']:3d} → {b['images_per_sec']:.0f} img/s | VRAM {b['vram_peak_gb']} GB\n")
    f.write(f"\nENTRAÎNEMENT ResNet50:\n")
    f.write(f"  {images_per_sec_train:.0f} img/s (batch 32) | VRAM max {vram_train} GB\n")

print(f"\n{'='*80}")
print("TEST TERMINÉ !")
print(f"Résultats sauvegardés dans {RESULTS_DIR}")
print("="*80)
