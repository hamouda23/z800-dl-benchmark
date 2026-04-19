# z800-dl-benchmark

Benchmark complet des performances Deep Learning sur une HP Z800 Workstation reconvertie en serveur IA headless.  
Stack : **TensorFlow 2.15 / Keras** — GPU : **NVIDIA Quadro P4000**

---

## Configuration du serveur

| Composant | Spécifications |
|-----------|---------------|
| **Machine** | HP Z800 Workstation (2009) |
| **CPU** | 2× Intel Xeon E5640 @ 2.67 GHz — 16 threads |
| **RAM** | 12–16 GB DDR3 ECC |
| **GPU** | NVIDIA Quadro P4000 — 8 GB GDDR5, 1792 CUDA cores |
| **OS** | Ubuntu Server 22.04 LTS, kernel 6.8.0 |
| **Wi-Fi** | Adaptateur USB Realtek RTL8192EU |
| **Stockage ML** | Disque dédié monté sur `/mnt/deep-learning/` |
| **TensorFlow** | 2.15.0 |
| **CUDA** | 12.x — Drivers NVIDIA 535+ |

---

## Structure du projet

```
.
├── benchmark_keras.py          # Benchmark général CPU vs GPU
├── test_keras_models.py        # ResNet50 inference + entraînement
├── test_keras_cv.py            # EfficientNetB0 (KerasCV)
├── test_keras_nlp.py           # BERT (KerasNLP)
├── test_mixed_precision.py     # FP32 vs FP16
├── test_timeseries_ecg.py      # Séries temporelles ECG (LSTM/GRU/CNN)
├── test_data_pipeline.py       # Benchmark pipelines tf.data
├── test_medical_imaging.py     # Classification + Segmentation médicale
├── test_transfer_learning.py   # Fine-tuning ResNet50/MobileNetV2/EfficientNetB3
└── results/                    # Résultats .txt et .json
```

---

## Scripts & Résultats

### `benchmark_keras.py` — Benchmark général CPU vs GPU

Compare les performances brutes entre CPU et GPU sur des opérations matricielles, Conv2D et blocs ResNet.

```
Matrice CPU : 18.09s
Matrice GPU : 0.98s   → ×18.52 plus rapide
Conv2D      : 1363 it/s
ResNet blk  : 249 it/s
VRAM pic    : 6.6 GB
```

---

### `test_keras_models.py` — ResNet50 Inference & Entraînement

Mesure le throughput de ResNet50 en inférence sur différents batch sizes, puis en entraînement.

```
INFERENCE ResNet50 :
  Batch  8  →  75 img/s | VRAM 0.19 GB
  Batch 16  → 151 img/s | VRAM 0.29 GB
  Batch 32  → 295 img/s | VRAM 0.46 GB
  Batch 64  → 294 img/s | VRAM 0.81 GB

ENTRAÎNEMENT :
  101 img/s (batch 32) | VRAM max 2.89 GB
```

---

### `test_keras_cv.py` — EfficientNetB0

Benchmark de l'architecture EfficientNetB0 en inférence image par image et en batch.

```
Single image : 5.46 FPS | Latence 183 ms
Batch 16     : 89 img/s | VRAM 0.03 GB
```

---

### `test_keras_nlp.py` — BERT base

Évalue les performances du modèle BERT pour le traitement du langage naturel.

```
Throughput : 3.94 it/s
Latence    : 253 ms/batch
VRAM       : 0.41 GB
```

---

### `test_mixed_precision.py` — FP32 vs FP16

Compare l'entraînement en précision simple (FP32) et demi-précision (FP16) sur la P4000.

```
FP32 : 101 img/s | VRAM 3.24 GB
FP16 : 107 img/s | VRAM 2.30 GB
Gain : +5.9% vitesse | −0.94 GB VRAM économisée
```

> La P4000 dispose de Tensor Cores limités, d'où un gain modeste en FP16.

---

### `test_timeseries_ecg.py` — Séries temporelles ECG

Simule la classification d'arythmies cardiaques (5 classes : Normal, FA, Bloc, Tachycardie, Bradycardie) sur des signaux ECG 12 dérivations avec 4 architectures.

| Modèle | Accuracy | Temps | VRAM | Params |
|--------|----------|-------|------|--------|
| 1D-CNN | 100.00% | 12.02s | 6.33 GB | 178,629 |
| LSTM | 100.00% | 51.16s | 0.36 GB | 126,085 |
| GRU | 100.00% | 50.21s | 0.38 GB | 96,261 |
| CNN+LSTM | 100.00% | 14.52s | 0.23 GB | 82,501 |

> **CNN+LSTM** offre le meilleur compromis : rapide (14s) et très économe en VRAM (0.23 GB).

---

### `test_data_pipeline.py` — Benchmark pipelines tf.data

Mesure l'impact de différentes optimisations de pipeline sur le débit CPU→GPU.

| Pipeline | img/s | Gain |
|----------|-------|------|
| 1. Naive | 902 | référence |
| 2. Prefetch | 992 | ×1.10 |
| 3. Shuffle + Prefetch | 1021 | ×1.13 |
| 4. Cache + Shuffle + Prefetch | 1009 | ×1.12 |
| 5. map parallèle + Prefetch | 479 | ×0.53 |

> Les gains sont faibles car les données sont en RAM. Avec lecture disque, le prefetch apporterait ×3 à ×10.

---

### `test_medical_imaging.py` — Imagerie médicale

**Classification radiographies pulmonaires** (Normal / Pneumonie / COVID-19) :

| Modèle | Accuracy | Throughput | VRAM |
|--------|----------|------------|------|
| MedCNN (léger) | 100.00% | 326 img/s | 2.16 GB |
| DenseNet121 | 93.50% | 38 img/s | 4.31 GB |

**Segmentation tumeur IRM (U-Net)** :

| Modèle | IoU | Pixel Accuracy | VRAM |
|--------|-----|----------------|------|
| U-Net | 0.9203 | 99.97% | 5.88 GB |

---

### `test_transfer_learning.py` — Transfer Learning & Fine-tuning

Fine-tuning en 2 phases sur un dataset médical 4 classes (Sain / Pneumonie / Tumeur / Fracture) :
- **Phase 1** : base gelée, seule la tête est entraînée (LR=1e-3)
- **Phase 2** : fine-tuning complet, base dégelée (LR=1e-4)

Architectures comparées : ResNet50 / MobileNetV2 / EfficientNetB3

> Résultats à venir après exécution.

---

## Conclusions générales

- Le GPU est **×18** plus rapide que le CPU sur les opérations matricielles
- La **VRAM de 8 GB** permet d'entraîner des modèles lourds (U-Net, DenseNet) sans OOM
- Le gain FP16 est modeste (+6%) car la P4000 a des Tensor Cores de génération ancienne
- **CNN+LSTM** est l'architecture optimale pour les séries temporelles sur ce hardware
- Pour les pipelines tf.data : `shuffle + prefetch` suffit quand les données sont en RAM

---

## Repos liés

- [ubuntu-server-network-rtl8192eu](https://github.com/hamouda23/ubuntu-server-network-rtl8192eu) — Configuration Wi-Fi RTL8192EU
- [z800-monitoring](https://github.com/hamouda23/z800-monitoring) — Monitoring Prometheus/Grafana
- [hp-z800-ai-agent](https://github.com/hamouda23/hp-z800-ai-agent) — Agent LLM Ollama natif
