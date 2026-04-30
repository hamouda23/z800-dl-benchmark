import torch
import torch.nn as nn
import torchvision.models as models
import time
import torch.cuda

print("="*80)
print("TEST PERFORMANCE IA - Quadro P4000 (explication pour débutant)")
print("="*80)

# Info carte
print(f"Carte : {torch.cuda.get_device_name(0)}")
print(f"VRAM totale : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} Go\n")

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

def benchmark_inference(model_name, batch_sizes=[16, 32, 64, 128]):
    print(f"\n--- INFERENCE {model_name.upper()} ---")
    model = getattr(models, model_name)(weights=None).cuda()
    model.eval()
    
    for bs in batch_sizes:
        try:
            x = torch.randn(bs, 3, 224, 224).cuda()
            
            # Warm-up
            for _ in range(10):
                with torch.no_grad():
                    _ = model(x)
            
            # Mesure réelle
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                with torch.no_grad():
                    _ = model(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            images_per_sec = (100 * bs) / elapsed
            vram = torch.cuda.max_memory_allocated() / 1024**3
            
            print(f"Batch {bs:3d} → {images_per_sec:6.0f} images/seconde | VRAM {vram:.1f} Go")
            
            torch.cuda.empty_cache()
        except RuntimeError:
            print(f"Batch {bs:3d} → Trop gros pour la VRAM (8 Go)")
            break

# Lancement des tests
benchmark_inference("resnet18")
benchmark_inference("resnet50", batch_sizes=[8, 16, 32, 64])  # ResNet50 plus gros

# ====================== TEST ENTRAINEMENT (Training) ======================
print("\n--- ENTRAINEMENT (Training) ResNet18 ---")
model = models.resnet18(weights=None).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

x = torch.randn(32, 3, 224, 224).cuda()
y = torch.randint(0, 1000, (32,)).cuda()

torch.cuda.reset_peak_memory_stats()
start = time.time()
for i in range(100):  # 100 itérations d'entraînement
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    if i == 0 or i == 99:
        torch.cuda.synchronize()
        print(f"Itération {i+1:3d} terminée")

elapsed_train = time.time() - start
images_per_sec_train = (100 * 32) / elapsed_train
vram_train = torch.cuda.max_memory_allocated() / 1024**3

print(f"\nENTRAÎNEMENT terminé :")
print(f"   {images_per_sec_train:.0f} images/seconde (batch 32)")
print(f"   VRAM max utilisée : {vram_train:.1f} Go")

print("\n" + "="*80)
print("✅ TEST TERMINÉ !")

print("="*80)