import tensorflow as tf
import numpy as np
import time
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
JSON_FILE = os.path.join(RESULTS_DIR, "test_data_pipeline.json")
TXT_FILE  = os.path.join(RESULTS_DIR, "test_data_pipeline.txt")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("="*70)
print("TEST PIPELINE tf.data — DÉBIT CPU→GPU")
print("="*70)
print(f"Date       : {timestamp}")
print(f"TensorFlow : {tf.__version__}")

N_SAMPLES  = 10_000
IMG_H, IMG_W, IMG_C = 224, 224, 3
N_CLASSES  = 10
BATCH_SIZE = 32
STEPS      = 100
AUTOTUNE   = tf.data.AUTOTUNE

results = {
    "timestamp": timestamp,
    "tensorflow": tf.__version__,
    "params": {"n_samples": N_SAMPLES, "image_size": f"{IMG_H}x{IMG_W}x{IMG_C}",
               "batch_size": BATCH_SIZE, "steps": STEPS},
    "pipelines": {}
}

# Données synthétiques en mémoire
X = np.random.randint(0, 256, (N_SAMPLES, IMG_H, IMG_W, IMG_C), dtype=np.uint8)
y = np.random.randint(0, N_CLASSES, N_SAMPLES)

# Modèle léger pour forcer le parcours du pipeline
def make_model():
    inp = tf.keras.Input(shape=(IMG_H, IMG_W, IMG_C))
    x = tf.keras.layers.GlobalAveragePooling2D()(inp)
    out = tf.keras.layers.Dense(N_CLASSES)(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return m


def benchmark_pipeline(ds, name, model):
    print(f"\n--- {name} ---")
    # warmup
    for batch in ds.take(3):
        pass
    start = time.time()
    for i, (xb, yb) in enumerate(ds):
        model.train_on_batch(xb, yb)
        if i + 1 >= STEPS:
            break
    elapsed = time.time() - start
    imgs_per_s = int(STEPS * BATCH_SIZE / elapsed)
    print(f"  {imgs_per_s} img/s  ({elapsed:.2f}s pour {STEPS} steps)")
    return {"images_per_s": imgs_per_s, "elapsed_s": round(elapsed, 2), "steps": STEPS}


print(f"\nImages : {N_SAMPLES} × {IMG_H}×{IMG_W}×{IMG_C} | Batch : {BATCH_SIZE} | Steps : {STEPS}")

# 1 — Sans optimisation (baseline)
model = make_model()
ds_naive = tf.data.Dataset.from_tensor_slices((X, y)).batch(BATCH_SIZE)
results["pipelines"]["naive"] = benchmark_pipeline(ds_naive, "1. Naive (sans optimisation)", model)

# 2 — prefetch seulement
model = make_model()
ds_prefetch = tf.data.Dataset.from_tensor_slices((X, y)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
results["pipelines"]["prefetch"] = benchmark_pipeline(ds_prefetch, "2. Prefetch (AUTOTUNE)", model)

# 3 — shuffle + prefetch
model = make_model()
ds_shuffle = (tf.data.Dataset.from_tensor_slices((X, y))
              .shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE))
results["pipelines"]["shuffle_prefetch"] = benchmark_pipeline(ds_shuffle, "3. Shuffle + Prefetch", model)

# 4 — cache + shuffle + prefetch
model = make_model()
ds_cache = (tf.data.Dataset.from_tensor_slices((X, y))
            .cache().shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE))
results["pipelines"]["cache_shuffle_prefetch"] = benchmark_pipeline(ds_cache, "4. Cache + Shuffle + Prefetch", model)

# 5 — num_parallel_calls sur map (normalisation float)
def normalize(x, label):
    return tf.cast(x, tf.float32) / 255.0, label

model = make_model()
ds_parallel = (tf.data.Dataset.from_tensor_slices((X, y))
               .map(normalize, num_parallel_calls=AUTOTUNE)
               .batch(BATCH_SIZE).prefetch(AUTOTUNE))
results["pipelines"]["parallel_map_prefetch"] = benchmark_pipeline(
    ds_parallel, "5. map parallèle + Prefetch (normalisation)", model)

# --- Résumé ---
print(f"\n{'='*70}")
print("RÉSUMÉ COMPARATIF")
print(f"{'='*70}")
print(f"{'Pipeline':<30} {'img/s':>8} {'Temps':>8} {'Gain vs naive':>14}")
print("-"*65)
naive_fps = results["pipelines"]["naive"]["images_per_s"]
labels = {
    "naive": "1. Naive",
    "prefetch": "2. Prefetch",
    "shuffle_prefetch": "3. Shuffle+Prefetch",
    "cache_shuffle_prefetch": "4. Cache+Shuffle+Prefetch",
    "parallel_map_prefetch": "5. map parallèle+Prefetch",
}
for key, label in labels.items():
    r = results["pipelines"][key]
    gain = f"×{r['images_per_s']/naive_fps:.2f}" if key != "naive" else "référence"
    print(f"{label:<30} {r['images_per_s']:>8} {r['elapsed_s']:>7.2f}s {gain:>14}")

# --- Sauvegarde ---
existing = []
if os.path.exists(JSON_FILE):
    with open(JSON_FILE) as f:
        existing = json.load(f)
existing.append(results)
with open(JSON_FILE, "w") as f:
    json.dump(existing, f, indent=2)

with open(TXT_FILE, "a") as f:
    f.write(f"\n{'='*70}\n")
    f.write(f"Run: {timestamp}\n")
    f.write(f"Batch: {BATCH_SIZE} | Steps: {STEPS} | Images: {N_SAMPLES}\n\n")
    f.write(f"{'Pipeline':<30} {'img/s':>8} {'Temps':>8} {'Gain vs naive':>14}\n")
    f.write("-"*65 + "\n")
    for key, label in labels.items():
        r = results["pipelines"][key]
        gain = f"×{r['images_per_s']/naive_fps:.2f}" if key != "naive" else "référence"
        f.write(f"{label:<30} {r['images_per_s']:>8} {r['elapsed_s']:>7.2f}s {gain:>14}\n")

print(f"\nRésultats sauvegardés dans {RESULTS_DIR}")
print("="*70)
