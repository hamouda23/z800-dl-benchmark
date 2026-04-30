import tensorflow as tf
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
JSON_FILE = os.path.join(RESULTS_DIR, "test1_tf.json")
TXT_FILE  = os.path.join(RESULTS_DIR, "test1_tf.txt")

gpus = tf.config.list_physical_devices('GPU')
gpu_available = len(gpus) > 0

print(f"TensorFlow: {tf.__version__}")
print(f"GPU disponible: {gpu_available}")
print(f"GPUs: {gpus}")

# --- Sauvegarde ---
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
result = {
    "timestamp": timestamp,
    "tensorflow_version": tf.__version__,
    "gpu_available": gpu_available,
    "gpus": [g.name for g in gpus]
}

# JSON
existing = []
if os.path.exists(JSON_FILE):
    with open(JSON_FILE) as f:
        existing = json.load(f)
existing.append(result)
with open(JSON_FILE, "w") as f:
    json.dump(existing, f, indent=2)

# TXT
with open(TXT_FILE, "a") as f:
    f.write(f"\n{'='*50}\n")
    f.write(f"Run: {timestamp}\n")
    f.write(f"TensorFlow : {tf.__version__}\n")
    f.write(f"GPU dispo  : {gpu_available}\n")
    f.write(f"GPUs       : {[g.name for g in gpus]}\n")

print(f"\nRésultats sauvegardés dans {RESULTS_DIR}")
