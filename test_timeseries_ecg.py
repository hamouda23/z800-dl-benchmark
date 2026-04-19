import tensorflow as tf
import numpy as np
import time
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
JSON_FILE = os.path.join(RESULTS_DIR, "test_timeseries_ecg.json")
TXT_FILE  = os.path.join(RESULTS_DIR, "test_timeseries_ecg.txt")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("="*70)
print("TEST SÉRIES TEMPORELLES ECG - LSTM/GRU/1D-CNN (Keras/TF)")
print("Simulation classification arythmie cardiaque")
print("="*70)
print(f"Date: {timestamp}")
print(f"TensorFlow: {tf.__version__}")

# Paramètres ECG
SEQUENCE_LEN  = 500    # ~2 secondes à 250 Hz
N_FEATURES    = 12     # 12 dérivations ECG standard
N_CLASSES     = 5      # Normal, FA, Bloc, Tachycardie, Bradycardie
BATCH_SIZE    = 64
EPOCHS        = 5
N_TRAIN       = 5000
N_TEST        = 1000

results = {"timestamp": timestamp, "tensorflow": tf.__version__,
           "ecg_params": {"sequence_len": SEQUENCE_LEN, "n_features": N_FEATURES,
                          "n_classes": N_CLASSES}, "models": {}}

print(f"\nParamètres ECG simulés :")
print(f"  Longueur séquence : {SEQUENCE_LEN} échantillons (~2s à 250Hz)")
print(f"  Dérivations       : {N_FEATURES} (ECG 12 dérivations)")
print(f"  Classes           : {N_CLASSES} (Normal, FA, Bloc, Tachy, Brady)")
print(f"  Dataset train     : {N_TRAIN} patients | test : {N_TEST} patients")


def generate_ecg_data(n_samples, seq_len, n_features, n_classes):
    """Simule des signaux ECG avec patterns par classe."""
    X = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
    y = np.random.randint(0, n_classes, n_samples)

    t = np.linspace(0, 2, seq_len)
    for i in range(n_samples):
        freq = [1.0, 1.5, 0.8, 2.5, 0.5][y[i]]     # fréquence cardiaque par classe
        noise = 0.1 * np.random.randn(seq_len, n_features)
        for f in range(n_features):
            X[i, :, f] = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t) + noise[:, f]

    return X, y


print("\nGénération des données ECG...")
X_train, y_train = generate_ecg_data(N_TRAIN, SEQUENCE_LEN, N_FEATURES, N_CLASSES)
X_test,  y_test  = generate_ecg_data(N_TEST,  SEQUENCE_LEN, N_FEATURES, N_CLASSES)
print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds  = tf.data.Dataset.from_tensor_slices((X_test,  y_test )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def train_and_evaluate(model, model_name):
    print(f"\n{'='*70}")
    print(f"MODÈLE : {model_name}")
    print(f"{'='*70}")
    print(f"  Paramètres : {model.count_params():,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    tf.config.experimental.reset_memory_stats('GPU:0')
    start = time.time()
    history = model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, verbose=1)
    elapsed = time.time() - start

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    mem = tf.config.experimental.get_memory_info('GPU:0')
    vram = round(mem['peak'] / 1024**3, 2)

    print(f"\n  Temps entraînement : {elapsed:.2f}s ({EPOCHS} epochs)")
    print(f"  Accuracy test      : {test_acc*100:.2f}%")
    print(f"  Loss test          : {test_loss:.4f}")
    print(f"  VRAM pic           : {vram} GB")

    return {
        "params": model.count_params(),
        "elapsed_seconds": round(elapsed, 2),
        "test_accuracy": round(test_acc * 100, 2),
        "test_loss": round(test_loss, 4),
        "vram_peak_gb": vram,
        "epochs": EPOCHS
    }


# --- Modèle 1 : 1D-CNN ---
def build_cnn1d():
    inp = tf.keras.Input(shape=(SEQUENCE_LEN, N_FEATURES))
    x = tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same')(inp)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(N_CLASSES)(x)
    return tf.keras.Model(inp, out)

results["models"]["1D-CNN"] = train_and_evaluate(build_cnn1d(), "1D-CNN")

# --- Modèle 2 : LSTM ---
def build_lstm():
    inp = tf.keras.Input(shape=(SEQUENCE_LEN, N_FEATURES))
    x = tf.keras.layers.LSTM(128, return_sequences=True)(inp)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(N_CLASSES)(x)
    return tf.keras.Model(inp, out)

results["models"]["LSTM"] = train_and_evaluate(build_lstm(), "LSTM")

# --- Modèle 3 : GRU ---
def build_gru():
    inp = tf.keras.Input(shape=(SEQUENCE_LEN, N_FEATURES))
    x = tf.keras.layers.GRU(128, return_sequences=True)(inp)
    x = tf.keras.layers.GRU(64)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(N_CLASSES)(x)
    return tf.keras.Model(inp, out)

results["models"]["GRU"] = train_and_evaluate(build_gru(), "GRU")

# --- Modèle 4 : CNN + LSTM hybride ---
def build_cnn_lstm():
    inp = tf.keras.Input(shape=(SEQUENCE_LEN, N_FEATURES))
    x = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inp)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(N_CLASSES)(x)
    return tf.keras.Model(inp, out)

results["models"]["CNN+LSTM"] = train_and_evaluate(build_cnn_lstm(), "CNN+LSTM (hybride)")

# --- Résumé comparatif ---
print(f"\n{'='*70}")
print("RÉSUMÉ COMPARATIF")
print(f"{'='*70}")
print(f"{'Modèle':<15} {'Accuracy':>10} {'Temps':>10} {'VRAM':>8} {'Params':>12}")
print("-"*60)
for name, r in results["models"].items():
    print(f"{name:<15} {r['test_accuracy']:>9.2f}% {r['elapsed_seconds']:>8.2f}s {r['vram_peak_gb']:>6.2f}GB {r['params']:>12,}")

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
    f.write(f"ECG: {SEQUENCE_LEN} samples, {N_FEATURES} dérivations, {N_CLASSES} classes\n\n")
    f.write(f"{'Modèle':<15} {'Accuracy':>10} {'Temps':>10} {'VRAM':>8} {'Params':>12}\n")
    f.write("-"*60 + "\n")
    for name, r in results["models"].items():
        f.write(f"{name:<15} {r['test_accuracy']:>9.2f}% {r['elapsed_seconds']:>8.2f}s "
                f"{r['vram_peak_gb']:>6.2f}GB {r['params']:>12,}\n")

print(f"\nRésultats sauvegardés dans {RESULTS_DIR}")
print("="*70)
