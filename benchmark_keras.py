import tensorflow as tf
import time
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
JSON_FILE = os.path.join(RESULTS_DIR, "benchmark_keras.json")
TXT_FILE  = os.path.join(RESULTS_DIR, "benchmark_keras.txt")

print("="*60)
print("BENCHMARK HP Z800 - NVIDIA Quadro P4000 (Keras/TF)")
print("="*60)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Date: {timestamp}")
print(f"TensorFlow: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
print(f"GPU: {gpus[0].name if gpus else 'Non détecté'}")

results = {"timestamp": timestamp, "tensorflow": tf.__version__}

# Test 1: Multiplication de matrices
print("\n" + "="*60)
print("TEST 1: Multiplication de matrices (10000x10000)")
print("="*60)
size = 10000

with tf.device('/CPU:0'):
    a = tf.random.normal([size, size])
    b = tf.random.normal([size, size])
    start = time.time()
    c = tf.matmul(a, b)
    _ = c.numpy()
    cpu_time = time.time() - start
print(f"\n--- CPU ---\nTemps: {cpu_time:.4f}s")

with tf.device('/GPU:0'):
    a = tf.random.normal([size, size])
    b = tf.random.normal([size, size])
    start = time.time()
    c = tf.matmul(a, b)
    _ = c.numpy()
    gpu_time = time.time() - start
speedup = cpu_time / gpu_time
print(f"--- GPU ---\nTemps: {gpu_time:.4f}s")
print(f"Accélération: {speedup:.2f}x")

results["matrix_10000x10000"] = {
    "cpu_seconds": round(cpu_time, 4),
    "gpu_seconds": round(gpu_time, 4),
    "speedup_x": round(speedup, 2)
}

# Test 2: Convolution 2D
print("\n" + "="*60)
print("TEST 2: Convolution 2D (simulation CNN)")
print("="*60)

conv_layer = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same')
input_data = tf.random.normal([128, 224, 224, 3])
_ = conv_layer(input_data)  # warm-up

start = time.time()
for _ in range(100):
    _ = conv_layer(input_data)
conv_time = time.time() - start
throughput_conv = round(100 / conv_time, 2)
print(f"100 itérations: {conv_time:.4f}s")
print(f"Throughput: {throughput_conv} it/s")

results["conv2d_100iter"] = {
    "seconds": round(conv_time, 4),
    "throughput_it_per_s": throughput_conv
}

# Test 3: Mémoire GPU
print("\n" + "="*60)
print("TEST 3: Informations mémoire GPU")
print("="*60)

mem_current = mem_peak = 0
if gpus:
    gpu_info = tf.config.experimental.get_memory_info('GPU:0')
    mem_current = round(gpu_info['current'] / 1024**3, 2)
    mem_peak    = round(gpu_info['peak']    / 1024**3, 2)
    print(f"Mémoire actuelle: {mem_current} GB")
    print(f"Mémoire pic:      {mem_peak} GB")

results["gpu_memory"] = {"current_gb": mem_current, "peak_gb": mem_peak}

# Test 4: Bloc ResNet
print("\n" + "="*60)
print("TEST 4: Simulation bloc ResNet")
print("="*60)

inputs = tf.keras.Input(shape=(56, 56, 64))
x = tf.keras.layers.Conv2D(64, 3, padding='same')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
model = tf.keras.Model(inputs, x)

data = tf.random.normal([32, 56, 56, 64])
_ = model(data, training=False)  # warm-up

start = time.time()
for _ in range(50):
    _ = model(data, training=False)
resnet_time = time.time() - start
throughput_resnet = round(50 / resnet_time, 2)
print(f"50 itérations: {resnet_time:.4f}s")
print(f"Throughput: {throughput_resnet} it/s")

results["resnet_block_50iter"] = {
    "seconds": round(resnet_time, 4),
    "throughput_it_per_s": throughput_resnet
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
    f.write(f"\n{'='*60}\n")
    f.write(f"Run: {timestamp}\n")
    f.write(f"TensorFlow : {tf.__version__}\n")
    f.write(f"Matrice CPU: {results['matrix_10000x10000']['cpu_seconds']}s\n")
    f.write(f"Matrice GPU: {results['matrix_10000x10000']['gpu_seconds']}s  (x{results['matrix_10000x10000']['speedup_x']})\n")
    f.write(f"Conv2D     : {throughput_conv} it/s\n")
    f.write(f"ResNet blk : {throughput_resnet} it/s\n")
    f.write(f"VRAM pic   : {mem_peak} GB\n")

print(f"\n{'='*60}")
print("BENCHMARK TERMINÉ")
print(f"Résultats sauvegardés dans {RESULTS_DIR}")
print("="*60)
