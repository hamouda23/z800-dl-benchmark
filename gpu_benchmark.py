import torch
import time
from datetime import datetime

print("="*60)
print("BENCHMARK HP Z800 - NVIDIA Quadro P4000")
print("="*60)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")

# Test 1: Multiplication de matrices
print("\n" + "="*60)
print("TEST 1: Multiplication de matrices (10000x10000)")
print("="*60)

size = 10000

# CPU
print("\n--- CPU ---")
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)
start = time.time()
c_cpu = torch.matmul(a_cpu, b_cpu)
cpu_time = time.time() - start
print(f"Temps: {cpu_time:.4f} secondes")

# GPU
print("\n--- GPU ---")
a_gpu = torch.randn(size, size).cuda()
b_gpu = torch.randn(size, size).cuda()
torch.cuda.synchronize()
start = time.time()
c_gpu = torch.matmul(a_gpu, b_gpu)
torch.cuda.synchronize()
gpu_time = time.time() - start
print(f"Temps: {gpu_time:.4f} secondes")
print(f"🚀 Accélération: {cpu_time/gpu_time:.2f}x")

# Test 2: Convolution 2D
print("\n" + "="*60)
print("TEST 2: Convolution 2D (simulation CNN)")
print("="*60)

batch_size = 128
conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda()
input_data = torch.randn(batch_size, 3, 224, 224).cuda()

torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    output = conv(input_data)
torch.cuda.synchronize()
conv_time = time.time() - start
print(f"100 itérations: {conv_time:.4f} secondes")
print(f"Throughput: {100/conv_time:.2f} it/s")

# Test 3: Utilisation mémoire
print("\n" + "="*60)
print("TEST 3: Utilisation mémoire GPU")
print("="*60)
print(f"Mémoire allouée: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"Mémoire réservée: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
print(f"Mémoire disponible: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")

# Test 4: CPU vs GPU - ResNet block
print("\n" + "="*60)
print("TEST 4: Simulation bloc ResNet")
print("="*60)

model = torch.nn.Sequential(
    torch.nn.Conv2d(64, 64, 3, padding=1),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 64, 3, padding=1),
    torch.nn.BatchNorm2d(64)
).cuda()

x = torch.randn(32, 64, 56, 56).cuda()

torch.cuda.synchronize()
start = time.time()
for _ in range(50):
    y = model(x)
torch.cuda.synchronize()
resnet_time = time.time() - start
print(f"50 itérations ResNet block: {resnet_time:.4f} secondes")
print(f"Throughput: {50/resnet_time:.2f} it/s")

print("\n" + "="*60)
print("✅ BENCHMARK TERMINÉ")
print("="*60)