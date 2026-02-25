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

print("Demo 83: Usando la libreria diffusers con DDPMPipeline para MNIST")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Tipo de dispositivo: ", device)

print("1. Cargando un Modelo de Difusion PreEntrenado de Digitos MNIST")
pipeline = DDPMPipeline.from_pretrained("1aurent/ddpm-mnist").to(device)

print("2. Generando 8 Imagenes de Digitos")
imagenesGeneradas = pipeline(batch_size=8).images

print("3. Ploteando los Digitos MNIST Generadas")
plotearImagenes(imagenesGeneradas, 2, 4, "Digitos MNIST con Diffusers")

print("4. Guardando la Primera Imagen Generada")
imagenesGeneradas[0].save("Digito.png")