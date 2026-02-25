import torch
import matplotlib.pyplot as plt
import torchvision

def plotearImagenes(imagenesGeneradas, filas, cols, titulo):
    for i in range(filas*cols):
        ax = plt.subplot(filas, cols, i + 1)
        img = imagenesGeneradas[i]/2+0.5 #-1,1
        img = (img.clamp(0, 1) * 255).byte() #0,1 => 0,255
        img = img.permute(1,2,0).cpu().numpy()
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.title(titulo)
    plt.subplots_adjust(hspace=-0.6)
    plt.show()

print("Demo 70: Usando Torch-Hub GAN Zoo Models - DCGAN")
use_gpu = True if torch.cuda.is_available() else False
print("GPU: ", use_gpu)
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub','DCGAN', pretrained=True, useGPU=use_gpu)
num_images = 8
ruido, _ = model.buildNoiseData(num_images)
print("Shape ruido: ", ruido.shape)
with torch.no_grad():
    imagenesGeneradas = model.test(ruido)
print("Shape imagenes generadas: ", imagenesGeneradas.shape)
plotearImagenes(imagenesGeneradas, 2, 4, "Torch-Hub DCGAN")
