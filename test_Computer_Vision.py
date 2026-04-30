
from ultralytics import YOLO
import torch
import time

print("="*50)
print("TEST YOLOV8 - COMPUTER VISION")
print("="*50)

# Charger YOLOv8
model = YOLO('yolov8m.pt')
model.to('cuda')

# Créer image test
img = torch.randint(0, 255, (1, 3, 640, 640)).float().cuda() / 255.0

# Warm-up
_ = model(img)

# Benchmark
start = time.time()
for _ in range(100):
    results = model(img, verbose=False)
torch.cuda.synchronize()
yolo_time = time.time() - start

print(f"YOLOv8m inférence:")
print(f"100 itérations: {yolo_time:.4f}s")
print(f"FPS: {100/yolo_time:.2f}")
print(f"Latence: {yolo_time/100*1000:.2f} ms/image")
