import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from diffusers import DDPMPipeline, DDIMScheduler
from datetime import datetime

horaInicio = datetime.now()
print("Demo 85: Usando la libreria diffusers con DDIMScheduler para CelebA")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Tipo de dispositivo: ", device)

print("1. Cargando un Modelo de Difusion PreEntrenado de Caras de CelebA")
scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
scheduler.set_timesteps(num_inference_steps=40)

print("2. Generando una Imagen de una Cara")
pipeline = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
pipeline.to(device)
pipeline.scheduler = scheduler
imagenesGeneradas = pipeline(num_inference_steps=40).images

horaFin = datetime.now()
tiempo = (horaFin - horaInicio).total_seconds()

print("3. Ploteando la Cara Generada")
img = imagenesGeneradas[0]
img.save("CaraDDIM.png")
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.suptitle("Cara de CelebA con DDIM")
plt.show()

print(f"Tiempo Total de Proceso DDIM: {tiempo} seg")