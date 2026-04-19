import tensorflow as tf
import numpy as np
import time
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
JSON_FILE = os.path.join(RESULTS_DIR, "test_medical_imaging.json")
TXT_FILE  = os.path.join(RESULTS_DIR, "test_medical_imaging.txt")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("="*70)
print("TEST IMAGERIE MÉDICALE — Classification & Segmentation (Keras/TF)")
print("Simulation radiographies pulmonaires + IRM cerveau")
print("="*70)
print(f"Date       : {timestamp}")
print(f"TensorFlow : {tf.__version__}")

# Paramètres
IMG_SIZE   = 256
BATCH_SIZE = 16
EPOCHS     = 5
N_TRAIN    = 2000
N_TEST     = 400
AUTOTUNE   = tf.data.AUTOTUNE

results = {
    "timestamp": timestamp,
    "tensorflow": tf.__version__,
    "params": {"img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
               "epochs": EPOCHS, "n_train": N_TRAIN, "n_test": N_TEST},
    "models": {}
}

print(f"\nParamètres :")
print(f"  Images      : {IMG_SIZE}×{IMG_SIZE} px (niveaux de gris)")
print(f"  Train/Test  : {N_TRAIN} / {N_TEST} images")
print(f"  Batch       : {BATCH_SIZE} | Epochs : {EPOCHS}")


# ── Données simulées ──────────────────────────────────────────────────────────

def make_classification_data(n, img_size, n_classes=3):
    """Simule des radiographies pulmonaires : Normal / Pneumonie / COVID."""
    X = np.random.randn(n, img_size, img_size, 1).astype(np.float32)
    y = np.random.randint(0, n_classes, n)
    for i in range(n):
        # Ajoute un pattern par classe (cercle/carré simulé)
        c = y[i]
        cx, cy = np.random.randint(60, 196, 2)
        r = [30, 45, 20][c]
        xx, yy = np.ogrid[:img_size, :img_size]
        mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
        X[i, mask, 0] += [0.8, 1.5, -0.5][c]
    return X, y

def make_segmentation_data(n, img_size):
    """Simule des IRM cerveau avec masque de tumeur."""
    X = np.random.randn(n, img_size, img_size, 1).astype(np.float32) * 0.3
    M = np.zeros((n, img_size, img_size, 1), dtype=np.float32)
    for i in range(n):
        cx, cy = np.random.randint(80, 176, 2)
        r = np.random.randint(15, 40)
        xx, yy = np.ogrid[:img_size, :img_size]
        mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
        X[i, mask, 0] += np.random.uniform(0.5, 1.5)
        M[i, mask, 0] = 1.0
    return X, M


def make_ds(X, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(500)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def train_eval(model, train_ds, test_ds, name, seg=False):
    print(f"\n{'='*70}\nMODÈLE : {name}\n{'='*70}")
    print(f"  Paramètres : {model.count_params():,}")

    loss_fn = ('binary_crossentropy' if seg
               else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    metrics = (['accuracy'] if not seg else
               [tf.keras.metrics.BinaryAccuracy(name='px_acc'),
                tf.keras.metrics.MeanIoU(num_classes=2, name='iou')])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=loss_fn, metrics=metrics)

    tf.config.experimental.reset_memory_stats('GPU:0')
    t0 = time.time()
    model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, verbose=1)
    elapsed = time.time() - t0

    ev = model.evaluate(test_ds, verbose=0)
    mem = tf.config.experimental.get_memory_info('GPU:0')
    vram = round(mem['peak'] / 1024**3, 2)

    if seg:
        loss_val, px_acc, iou = ev
        print(f"\n  Temps         : {elapsed:.2f}s")
        print(f"  Pixel accuracy: {px_acc*100:.2f}%")
        print(f"  IoU           : {iou:.4f}")
        print(f"  VRAM pic      : {vram} GB")
        return {"params": model.count_params(), "elapsed_s": round(elapsed, 2),
                "pixel_accuracy": round(px_acc*100, 2), "iou": round(float(iou), 4),
                "vram_peak_gb": vram}
    else:
        loss_val, acc = ev
        imgs_per_s = int(N_TRAIN * EPOCHS / elapsed)
        print(f"\n  Temps         : {elapsed:.2f}s")
        print(f"  Accuracy      : {acc*100:.2f}%")
        print(f"  Throughput    : {imgs_per_s} img/s")
        print(f"  VRAM pic      : {vram} GB")
        return {"params": model.count_params(), "elapsed_s": round(elapsed, 2),
                "accuracy": round(acc*100, 2), "throughput_img_s": imgs_per_s,
                "vram_peak_gb": vram}


# ── PARTIE 1 : Classification radiographies ───────────────────────────────────

print("\n" + "="*70)
print("PARTIE 1 — Classification radiographies pulmonaires (3 classes)")
print("           Normal / Pneumonie / COVID-19")
print("="*70)

X_tr, y_tr = make_classification_data(N_TRAIN, IMG_SIZE)
X_te, y_te = make_classification_data(N_TEST,  IMG_SIZE)
tr_ds = make_ds(X_tr, y_tr, shuffle=True)
te_ds = make_ds(X_te, y_te)

# Modèle A : CNN médical léger
def build_med_cnn():
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(3)(x)
    return tf.keras.Model(inp, out)

results["models"]["MedCNN"] = train_eval(build_med_cnn(), tr_ds, te_ds, "CNN médical léger")

# Modèle B : Transfer learning DenseNet121 (architecture standard en médical)
def build_densenet():
    base = tf.keras.applications.DenseNet121(
        include_top=False, weights=None,
        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = tf.keras.layers.Concatenate()([inp, inp, inp])  # gris → 3 canaux
    x = base(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(3)(x)
    return tf.keras.Model(inp, out)

results["models"]["DenseNet121"] = train_eval(build_densenet(), tr_ds, te_ds, "DenseNet121 (standard médical)")


# ── PARTIE 2 : Segmentation IRM (U-Net simplifié) ────────────────────────────

print("\n" + "="*70)
print("PARTIE 2 — Segmentation tumeur IRM cerveau (U-Net)")
print("="*70)

X_seg_tr, M_tr = make_segmentation_data(N_TRAIN, IMG_SIZE)
X_seg_te, M_te = make_segmentation_data(N_TEST,  IMG_SIZE)
seg_tr = make_ds(X_seg_tr, M_tr, shuffle=True)
seg_te = make_ds(X_seg_te, M_te)

def build_unet():
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    # Encoder
    c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D(2)(c1)

    c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D(2)(c2)

    # Bottleneck
    b = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    b = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(b)

    # Decoder
    u1 = tf.keras.layers.UpSampling2D(2)(b)
    u1 = tf.keras.layers.Concatenate()([u1, c2])
    c3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    c3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c3)

    u2 = tf.keras.layers.UpSampling2D(2)(c3)
    u2 = tf.keras.layers.Concatenate()([u2, c1])
    c4 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(u2)
    c4 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c4)

    out = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c4)
    return tf.keras.Model(inp, out)

results["models"]["UNet"] = train_eval(build_unet(), seg_tr, seg_te, "U-Net segmentation tumeur", seg=True)


# ── Résumé ────────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("RÉSUMÉ")
print(f"{'='*70}")
print(f"\nClassification pulmonaire :")
for name in ["MedCNN", "DenseNet121"]:
    r = results["models"][name]
    print(f"  {name:<15} acc={r['accuracy']:>6.2f}%  {r['throughput_img_s']:>5} img/s  VRAM {r['vram_peak_gb']} GB")
print(f"\nSegmentation IRM :")
r = results["models"]["UNet"]
print(f"  {'U-Net':<15} IoU={r['iou']:.4f}  px_acc={r['pixel_accuracy']:.2f}%  VRAM {r['vram_peak_gb']} GB")

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
    f.write(f"Images: {IMG_SIZE}×{IMG_SIZE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}\n\n")
    f.write("Classification pulmonaire :\n")
    for name in ["MedCNN", "DenseNet121"]:
        r = results["models"][name]
        f.write(f"  {name:<15} acc={r['accuracy']:>6.2f}%  {r['throughput_img_s']:>5} img/s  VRAM {r['vram_peak_gb']} GB\n")
    f.write("\nSegmentation IRM :\n")
    r = results["models"]["UNet"]
    f.write(f"  {'U-Net':<15} IoU={r['iou']:.4f}  px_acc={r['pixel_accuracy']:.2f}%  VRAM {r['vram_peak_gb']} GB\n")

print(f"\nRésultats sauvegardés dans {RESULTS_DIR}")
print("="*70)
