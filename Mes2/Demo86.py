import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from diffusers import DDPMPipeline, DDIMScheduler
from datetime import datetime

horaInicio = datetime.now()
print("Demo 86: Usando la libreria diffusers con DDIMScheduler para generar Archivos con Caras")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Tipo de dispositivo: ", device)

nCaras = int(input("Ingresa el Numero de Caras a Generar y Guardar en Disco: "))

print("1. Cargando un Modelo de Difusion PreEntrenado de Caras de CelebA")
scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
scheduler.set_timesteps(num_inference_steps=40)

print("2. Generando una Imagen de una Cara")
pipeline = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
pipeline.to(device)
pipeline.scheduler = scheduler

nBloques = int(nCaras / 2)
if(nCaras % 2>0):
    nBloques += 1
x = 0
ruta = "datasets/CarasDDIM/"
for i in range(nBloques):
    nMuestras = 2
    if((i==(nBloques-1)) and (nCaras % 2>0)):
        nMuestras = 1
    imagenesGeneradas = pipeline(num_inference_steps=40, batch_size=nMuestras).images
    for j in range(nMuestras):
        x += 1
        img = imagenesGeneradas[j]
        archivo = ruta + str(x) + ".png"
        img.save(archivo)

horaFin = datetime.now()
tiempo = (horaFin - horaInicio).total_seconds()
print(f"Tiempo Total de Proceso DDIM: {tiempo} seg")
print(f"Total de Archivos con Caras creadas: {x}")