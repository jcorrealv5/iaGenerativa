import torch
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim

def plotearImagenes(imagenesGeneradas, filas, cols, titulo):
    fig, axes = plt.subplots(filas, cols)
    for i in range(filas*cols):
        row = i // cols
        col = i % cols
        img = imagenesGeneradas[i]/2+0.5 #-1,1
        img = (img.clamp(0, 1) * 255).byte() #0,1 => 0,255
        img = img.permute(1,2,0).cpu().numpy()
        axes[row, col].imshow(img)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    plt.title(titulo)
    plt.show()

print("Demo 72: Usando Torch-Hub GAN Zoo Models - PGAN")
use_gpu = True if torch.cuda.is_available() else False
print("GPU: ", use_gpu)
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub','PGAN', model_name='celebAHQ-512', pretrained=True, useGPU=use_gpu)
num_images = 8
ruido, _ = model.buildNoiseData(num_images)
print("Shape ruido: ", ruido.shape)
with torch.no_grad():
    imagenesGeneradas = model.test(ruido)
print("Shape imagenes generadas: ", imagenesGeneradas.shape)
plotearImagenes(imagenesGeneradas, 2, 4, "Torch-Hub PGAN")
