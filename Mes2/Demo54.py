import torch.nn as nn
import torch, os
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
import albumentations 
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from modGAN import LoadData, Generador

print("Demo 54: Prueba del Modelo CycleGAN de Sexo")

batch_size=1
print("1. Cargar el DataSet y DataLoader de CelebA")
transforms = albumentations.Compose(
[albumentations.Resize(width=256, height=256),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Normalize(mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],max_pixel_value=255),
    ToTensorV2()],
additional_targets={"image0": "image"})
dataset = LoadData(root_A=["datasets/Sexo/Femenino/"],root_B=["datasets/Sexo/Masculino/"],transform=transforms)
loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

def plotearCaras(imagenesGeneradas, titulo):
    plt.figure(figsize=(5, 5))
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        img = imagenesGeneradas[i]/2+0.5 #-1,1
        img = (img.clamp(0, 1) * 255).byte() #0,1 => 0,255
        img = img.permute(1,2,0).cpu().numpy()
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.title(titulo)
    plt.subplots_adjust(hspace=-0.6)
    plt.show()

print("2. Generando Caras con Pelos Negros y Rubios")
device = "cuda" if torch.cuda.is_available() else "cpu"
gen_A = Generador(img_channels=3, num_residuals=9).to(device)
gen_B = Generador(img_channels=3, num_residuals=9).to(device)
gen_A.load_state_dict(torch.load("preentrenados/GAN/Sexo/gen_mujer_1.pth"))
gen_B.load_state_dict(torch.load("preentrenados/GAN/Sexo/gen_hombre_1.pth"))
i=0
for mujer,hombre in loader:
    i=i+1
    print(f"Procesando imagen: {i}")
    imagenesGeneradas = []    
    imagenesGeneradas.append(mujer.squeeze(0))
    fake_hombre=gen_B(mujer.to(device)).squeeze(0)
    imagenesGeneradas.append(fake_hombre)
    fake2mujer=gen_A(fake_hombre).squeeze(0)
    imagenesGeneradas.append(fake2mujer)
    imagenesGeneradas.append(hombre.squeeze(0))
    fake_mujer=gen_A(hombre.to(device)).squeeze(0)
    imagenesGeneradas.append(fake_mujer)
    fake2hombre=gen_B(fake_mujer).squeeze(0)
    imagenesGeneradas.append(fake2hombre)
    print("imagenesGeneradas: ", len(imagenesGeneradas))
    plotearCaras(imagenesGeneradas, "Serie: " + str(i))
    if i>4:
        break