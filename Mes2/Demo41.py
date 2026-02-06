import torch
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os, cv2

print("Demo 41: Generando Caras de Alumnos en Disco usando cGAN")
device="cuda" if torch.cuda.is_available() else "cpu"

archivo = "preentrenados/GAN/Alumnos/cGAN_Alumnos_1594_0.08066758513450623.pt"
generador=torch.jit.load(archivo, map_location=device)
generador.eval()

idAlumno = int(input("Ingresa el id del Alumno de 0 a 4 a generar: "))
batch_size = int(input("Cuantos archivos deseas generar con caras: "))

z = Variable(torch.randn(batch_size, 100)).cuda()
labels = torch.LongTensor([idAlumno for i in range(batch_size)]).cuda()
imagenesGeneradas = generador(z, labels).cpu().detach()

carpeta = str(idAlumno)
if(not os.path.isdir(carpeta)):
    os.makedirs(carpeta)
c=0
for i in range(batch_size):
    c = c + 1
    print("Creando imagen: " + str(c))
    img = imagenesGeneradas[i]/2+0.5 #-1,1
    img = (img.clamp(0, 1) * 255).byte() #0,1 => 0,255
    img = img.permute(1,2,0).cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img,(200,200))
    archivo = os.path.join(carpeta, str(c) + ".png")
    cv2.imwrite(archivo, img)
print(f"Se crearon: {batch_size} archivos con caras del alumno: {idAlumno}")