# Projet : Benchmark Deep Learning — HP Z800

## Serveur

| Composant | Specs |
|-----------|-------|
| **Machine** | HP Z800 Workstation (2009) |
| **CPU** | 2× Intel Xeon E5640 @ 2.67 GHz — 16 threads |
| **RAM** | 12–16 GB DDR3 ECC |
| **GPU** | NVIDIA Quadro P4000 — 8 GB GDDR5, 1792 CUDA cores |
| **OS** | Ubuntu Server 22.04 LTS, kernel 6.8.0 |
| **Wi-Fi** | Adaptateur USB Realtek RTL8192EU (pilote DKMS) |
| **Stockage** | `/mnt/deep-learning/` — disque dédié Conda + modèles ML |

**GitHub de référence :**
- Configuration Wi-Fi : https://github.com/hamouda23/ubuntu-server-network-rtl8192eu
- Monitoring Prometheus/Grafana : https://github.com/hamouda23/z800-monitoring
- Agent LLM Ollama natif : https://github.com/hamouda23/hp-z800-ai-agent

---

## Objectif du projet

Benchmarker les performances du serveur pour le Deep Learning avec **TensorFlow / Keras**.
Chaque script de test sauvegarde ses résultats dans `results/` en format `.txt` et `.json`.

---

## Dossier de travail

```
/mnt/deep-learning/first-test/
├── results/                  ← résultats de tous les tests
├── benchmark_keras.py        ← benchmark général Keras (CPU vs GPU, Conv2D, ResNet)
├── test1_tf.py               ← test basique TF/GPU
├── test_keras_models.py      ← ResNet50 inference + entraînement
├── test_keras_cv.py          ← EfficientNetB0 (KerasCV)
├── test_keras_nlp.py         ← BERT (KerasNLP)
├── test_mixed_precision.py   ← FP32 vs FP16
├── test_timeseries_ecg.py    ← LSTM/GRU/1D-CNN sur données ECG (en cours)
├── test_Computer_Vision.py   ← YOLO (script fourni par l'utilisateur, PyTorch)
├── test_Pytorch.py           ← PyTorch (script fourni par l'utilisateur)
├── test_Transformers.py      ← HuggingFace (script fourni par l'utilisateur)
└── gpu_benchmark.py          ← GPU benchmark PyTorch (script fourni par l'utilisateur)
```

> **Note :** Les scripts PyTorch/YOLO/HuggingFace ont été fournis par l'utilisateur.
> Les scripts TensorFlow/Keras ont été écrits par Claude.

---

## Résultats obtenus (TensorFlow 2.15.0)

### benchmark_keras.py
- CPU matrice : 18.09s | GPU matrice : 0.98s → **×18.52 plus rapide**
- Conv2D : 1363 it/s | ResNet block : 249 it/s
- VRAM pic : 6.6 GB

### test_keras_models.py — ResNet50
| Mode | Batch | Perf | VRAM |
|------|-------|------|------|
| Inference | 8 | 75 img/s | 0.19 GB |
| Inference | 32 | 295 img/s | 0.46 GB |
| Entraînement | 32 | 101 img/s | 2.89 GB |

### test_keras_cv.py — EfficientNetB0
- Single image : 5.46 FPS, latence 183 ms
- Batch 16 : 89 img/s, VRAM 0.03 GB

### test_keras_nlp.py — BERT base
- Throughput : 3.94 it/s, latence 253 ms/batch, VRAM 0.41 GB

### test_mixed_precision.py — FP32 vs FP16
- FP32 : 101 img/s, 3.24 GB VRAM
- FP16 : 107 img/s, 2.30 GB VRAM → **+5.9% vitesse, −0.94 GB VRAM**

---

## Test en cours : test_timeseries_ecg.py

Simulation classification arythmie cardiaque avec 4 architectures :
- **1D-CNN** : Conv1D multicouche
- **LSTM** : 2 couches LSTM empilées
- **GRU** : 2 couches GRU empilées
- **CNN+LSTM** : hybride Conv1D + LSTM

Paramètres : séquences 500 échantillons (≈2s à 250 Hz), 12 dérivations ECG, 5 classes, 5000 patients train, 1000 test, batch 64, 5 epochs.
Résultats → `results/test_timeseries_ecg.txt` et `results/test_timeseries_ecg.json`

---

## Convention pour les scripts

- Résultats toujours sauvegardés dans `results/` en `.txt` **et** `.json`
- Format `.txt` : lisible humain avec `Run: YYYY-MM-DD HH:MM:SS`
- Format `.json` : tableau d'objets (append à chaque run)
- Mesurer systématiquement : throughput, latence, VRAM, temps total
