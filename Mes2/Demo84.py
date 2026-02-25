import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from diffusers import DDPMPipeline

print("Demo 84: Usando la libreria diffusers con DDPMPipeline para CelebA")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Tipo de dispositivo: ", device)

print("1. Cargando un Modelo de Difusion PreEntrenado de Caras de CelebA")
pipeline = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256").to(device)

print("2. Generando una Imagen de una Cara")
imagenesGeneradas = pipeline(batch_size=1).images

print("3. Ploteando la Cara Generada")
img = imagenesGeneradas[0]
img.save("Cara.png")
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.suptitle("Cara de CelebA con Diffusers")
plt.show()