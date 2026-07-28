import torch

print("=== VERIFICA GPU DA MAC A WINDOWS ===")
print("GPU disponibile: ", torch.cuda.is_available())
if(torch.cuda.is_available()):
    print("GPU in uso: ", torch.cuda.get_device_name(0))
else:
    print("Nessuna GPU")