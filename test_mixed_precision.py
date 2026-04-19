import tensorflow as tf
from tensorflow.keras import mixed_precision
import time
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
JSON_FILE = os.path.join(RESULTS_DIR, "test_mixed_precision.json")
TXT_FILE  = os.path.join(RESULTS_DIR, "test_mixed_precision.txt")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("="*70)
print("TEST MIXED PRECISION - FP32 vs FP16 (Quadro P4000)")
print("="*70)
print(f"Date: {timestamp}")
print(f"TensorFlow: {tf.__version__}")

results = {"timestamp": timestamp, "tensorflow": tf.__version__, "tests": {}}


def build_model(input_shape=(224, 224, 3), num_classes=1000):
    base = tf.keras.applications.ResNet50(
        weights=None, include_top=False, input_shape=input_shape)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(base.input, x)


def run_benchmark(precision_label, iterations=100, batch_size=32):
    print(f"\n{'='*70}")
    print(f"MODE: {precision_label}")
    print(f"{'='*70}")

    model = build_model()
    optimizer = tf.keras.optimizers.Adam(0.001)

    if precision_label == "FP16":
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    dtype = tf.float16 if precision_label == "FP16" else tf.float32
    x = tf.cast(tf.random.normal([batch_size, 224, 224, 3]), dtype)
    y = tf.random.uniform([batch_size], minval=0, maxval=1000, dtype=tf.int32)

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            logits = model(xb, training=True)
            logits = tf.cast(logits, tf.float32)
            loss = loss_fn(yb, logits)
            if precision_label == "FP16":
                loss = optimizer.get_scaled_loss(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        if precision_label == "FP16":
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Warm-up
    for _ in range(5):
        train_step(x, y)

    tf.config.experimental.reset_memory_stats('GPU:0')
    start = time.time()
    for i in range(iterations):
        loss = train_step(x, y)
        if i == 0 or i == iterations - 1:
            print(f"  Itération {i+1:3d} — loss: {float(loss):.4f}")

    elapsed = time.time() - start
    images_per_sec = round((iterations * batch_size) / elapsed, 0)
    mem = tf.config.experimental.get_memory_info('GPU:0')
    vram_peak = round(mem['peak'] / 1024**3, 2)

    print(f"\n  Temps total  : {elapsed:.2f}s")
    print(f"  Images/s     : {images_per_sec:.0f}")
    print(f"  VRAM pic     : {vram_peak} GB")

    return {
        "images_per_sec": images_per_sec,
        "elapsed_seconds": round(elapsed, 2),
        "vram_peak_gb": vram_peak,
        "batch_size": batch_size,
        "iterations": iterations
    }


# --- FP32 ---
results["tests"]["FP32"] = run_benchmark("FP32")

# --- FP16 ---
mixed_precision.set_global_policy('mixed_float16')
results["tests"]["FP16"] = run_benchmark("FP16")
mixed_precision.set_global_policy('float32')  # reset

# --- Comparaison ---
fp32 = results["tests"]["FP32"]["images_per_sec"]
fp16 = results["tests"]["FP16"]["images_per_sec"]
gain = round((fp16 - fp32) / fp32 * 100, 1)
vram_saved = round(
    results["tests"]["FP32"]["vram_peak_gb"] -
    results["tests"]["FP16"]["vram_peak_gb"], 2)

results["comparison"] = {
    "speedup_percent": gain,
    "vram_saved_gb": vram_saved
}

print(f"\n{'='*70}")
print("COMPARAISON FP32 vs FP16")
print(f"{'='*70}")
print(f"  FP32         : {fp32:.0f} img/s  | VRAM {results['tests']['FP32']['vram_peak_gb']} GB")
print(f"  FP16         : {fp16:.0f} img/s  | VRAM {results['tests']['FP16']['vram_peak_gb']} GB")
print(f"  Gain vitesse : {gain:+.1f}%")
print(f"  VRAM économisée : {vram_saved} GB")

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
    f.write(f"FP32         : {fp32:.0f} img/s | VRAM {results['tests']['FP32']['vram_peak_gb']} GB\n")
    f.write(f"FP16         : {fp16:.0f} img/s | VRAM {results['tests']['FP16']['vram_peak_gb']} GB\n")
    f.write(f"Gain vitesse : {gain:+.1f}%\n")
    f.write(f"VRAM économisée : {vram_saved} GB\n")

print(f"\nRésultats sauvegardés dans {RESULTS_DIR}")
print("="*70)
