import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from diffusers import DDPMPipeline

def plotearImagenes(imagenesGeneradas, filas, cols, titulo):
    for i in range(filas*cols):
        ax = plt.subplot(filas, cols, i + 1)
        img = imagenesGeneradas[i]
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.suptitle(titulo)
    plt.subplots_adjust(hspace=-0.6)
    plt.show()

print("Demo 82: Usando la libreria diffusers con DDPMPipeline para Mariposas")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Tipo de dispositivo: ", device)

print("1. Cargando un Modelo de Difusion PreEntrenado de Mariposas")
pipeline = DDPMPipeline.from_pretrained("johnowhitaker/ddpm-butterflies-32px").to(device)

print("2. Generando 8 Imagenes de Mariposas")
imagenesGeneradas = pipeline(batch_size=8).images

print("3. Ploteando las Mariposas Generadas")
plotearImagenes(imagenesGeneradas, 2, 4, "Mariposas con Diffusers")