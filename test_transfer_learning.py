import tensorflow as tf
import numpy as np
import time
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
JSON_FILE = os.path.join(RESULTS_DIR, "test_transfer_learning.json")
TXT_FILE  = os.path.join(RESULTS_DIR, "test_transfer_learning.txt")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("="*70)
print("TEST TRANSFER LEARNING — Fine-tuning sur dataset custom (Keras/TF)")
print("Simulation classification médicale 4 classes")
print("="*70)
print(f"Date       : {timestamp}")
print(f"TensorFlow : {tf.__version__}")

IMG_SIZE   = 224
BATCH_SIZE = 32
N_TRAIN    = 3000
N_TEST     = 600
N_CLASSES  = 4
AUTOTUNE   = tf.data.AUTOTUNE

results = {
    "timestamp": timestamp,
    "tensorflow": tf.__version__,
    "params": {"img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
               "n_train": N_TRAIN, "n_test": N_TEST, "n_classes": N_CLASSES},
    "models": {}
}

print(f"\nDataset : {N_TRAIN} train / {N_TEST} test | {N_CLASSES} classes | {IMG_SIZE}×{IMG_SIZE} RGB")
print(f"Classes : Sain / Pneumonie / Tumeur / Fracture")


# ── Données simulées ──────────────────────────────────────────────────────────

def make_data(n, img_size, n_classes):
    X = np.random.randn(n, img_size, img_size, 3).astype(np.float32) * 0.3
    y = np.random.randint(0, n_classes, n)
    for i in range(n):
        cx, cy = np.random.randint(50, img_size - 50, 2)
        r = [0, 35, 50, 20][y[i]]
        if r > 0:
            xx, yy = np.ogrid[:img_size, :img_size]
            mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
            X[i, mask, y[i] % 3] += np.random.uniform(0.6, 1.2)
    # Normalisation ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    X = (X - mean) / std
    return X, y

def make_ds(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(500)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("\nGénération des données...")
X_tr, y_tr = make_data(N_TRAIN, IMG_SIZE, N_CLASSES)
X_te, y_te = make_data(N_TEST,  IMG_SIZE, N_CLASSES)
tr_ds = make_ds(X_tr, y_tr, shuffle=True)
te_ds = make_ds(X_te, y_te)


# ── Entraînement ──────────────────────────────────────────────────────────────

def train_phase(model, ds_tr, ds_te, epochs, label):
    t0 = time.time()
    h = model.fit(ds_tr, epochs=epochs, validation_data=ds_te, verbose=1)
    elapsed = time.time() - t0
    _, acc = model.evaluate(ds_te, verbose=0)
    return round(elapsed, 2), round(acc * 100, 2), h.history


def benchmark(base_fn, name, freeze_epochs=3, finetune_epochs=5):
    print(f"\n{'='*70}\nMODÈLE : {name}\n{'='*70}")

    base = base_fn(include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(N_CLASSES)(x)
    model = tf.keras.Model(inp, out)

    total_params = model.count_params()
    print(f"  Paramètres totaux : {total_params:,}")

    # ── Phase 1 : tête seulement (base gelée) ────────────────────────────────
    print(f"\n  Phase 1 — Tête seulement (base gelée) — {freeze_epochs} epochs")
    base.trainable = False
    trainable_frozen = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"  Paramètres entraînés : {trainable_frozen:,}")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    tf.config.experimental.reset_memory_stats('GPU:0')
    t_freeze, acc_freeze, _ = train_phase(model, tr_ds, te_ds, freeze_epochs, "freeze")
    mem_freeze = round(tf.config.experimental.get_memory_info('GPU:0')['peak'] / 1024**3, 2)

    # ── Phase 2 : fine-tuning (base dégelée) ─────────────────────────────────
    print(f"\n  Phase 2 — Fine-tuning complet (base dégelée) — {finetune_epochs} epochs")
    base.trainable = True
    trainable_full = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"  Paramètres entraînés : {trainable_full:,}")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),  # LR réduit
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    tf.config.experimental.reset_memory_stats('GPU:0')
    t_finetune, acc_finetune, _ = train_phase(model, tr_ds, te_ds, finetune_epochs, "finetune")
    mem_finetune = round(tf.config.experimental.get_memory_info('GPU:0')['peak'] / 1024**3, 2)

    imgs_per_s = int(N_TRAIN * finetune_epochs / t_finetune)

    print(f"\n  Résumé {name}:")
    print(f"    Phase 1 (gelé)    : acc={acc_freeze:.2f}%  temps={t_freeze}s  VRAM={mem_freeze} GB")
    print(f"    Phase 2 (fin-tun) : acc={acc_finetune:.2f}%  temps={t_finetune}s  VRAM={mem_finetune} GB")
    print(f"    Gain accuracy     : +{acc_finetune - acc_freeze:.2f}%")
    print(f"    Throughput        : {imgs_per_s} img/s")

    return {
        "total_params": total_params,
        "phase1_frozen": {
            "trainable_params": int(trainable_frozen),
            "elapsed_s": t_freeze,
            "accuracy": acc_freeze,
            "vram_peak_gb": mem_freeze,
            "epochs": freeze_epochs
        },
        "phase2_finetune": {
            "trainable_params": int(trainable_full),
            "elapsed_s": t_finetune,
            "accuracy": acc_finetune,
            "vram_peak_gb": mem_finetune,
            "epochs": finetune_epochs,
            "throughput_img_s": imgs_per_s
        },
        "accuracy_gain": round(acc_finetune - acc_freeze, 2)
    }


results["models"]["ResNet50"]   = benchmark(tf.keras.applications.ResNet50,   "ResNet50")
results["models"]["MobileNetV2"] = benchmark(tf.keras.applications.MobileNetV2, "MobileNetV2")
results["models"]["EfficientNetB3"] = benchmark(tf.keras.applications.EfficientNetB3, "EfficientNetB3")


# ── Résumé comparatif ─────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("RÉSUMÉ COMPARATIF — Fine-tuning")
print(f"{'='*70}")
print(f"{'Modèle':<18} {'Acc gelé':>9} {'Acc finetune':>13} {'Gain':>7} {'img/s':>7} {'VRAM':>7} {'Params':>12}")
print("-"*80)
for name, r in results["models"].items():
    p1 = r["phase1_frozen"]
    p2 = r["phase2_finetune"]
    print(f"{name:<18} {p1['accuracy']:>8.2f}% {p2['accuracy']:>12.2f}% "
          f"{r['accuracy_gain']:>+6.2f}% {p2['throughput_img_s']:>7} "
          f"{p2['vram_peak_gb']:>6.2f}GB {r['total_params']:>12,}")


# ── Sauvegarde ────────────────────────────────────────────────────────────────

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
    f.write(f"Dataset: {N_TRAIN} train / {N_TEST} test | {N_CLASSES} classes | {IMG_SIZE}×{IMG_SIZE}\n\n")
    f.write(f"{'Modèle':<18} {'Acc gelé':>9} {'Acc finetune':>13} {'Gain':>7} {'img/s':>7} {'VRAM':>7} {'Params':>12}\n")
    f.write("-"*80 + "\n")
    for name, r in results["models"].items():
        p1 = r["phase1_frozen"]
        p2 = r["phase2_finetune"]
        f.write(f"{name:<18} {p1['accuracy']:>8.2f}% {p2['accuracy']:>12.2f}% "
                f"{r['accuracy_gain']:>+6.2f}% {p2['throughput_img_s']:>7} "
                f"{p2['vram_peak_gb']:>6.2f}GB {r['total_params']:>12,}\n")

print(f"\nRésultats sauvegardés dans {RESULTS_DIR}")
print("="*70)
