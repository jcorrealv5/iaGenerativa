import torch
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

print("Demo 40: Probando la Generacion de Caras de Alumnos con cGAN")
device="cuda" if torch.cuda.is_available() else "cpu"

archivo = "preentrenados/GAN/Alumnos/cGAN_Alumnos_1594_0.08066758513450623.pt"
generador=torch.jit.load(archivo, map_location=device)
generador.eval()

batch_size = 30
z = Variable(torch.randn(batch_size, 100)).cuda()
labels = torch.LongTensor([i for i in range(5) for _ in range(6)]).cuda()
imagenesGeneradas = generador(z, labels).cpu().detach()
print(imagenesGeneradas.shape)
for i in range(batch_size):
    ax = plt.subplot(5, 6, i + 1)
    img = imagenesGeneradas[i]/2+0.5 #-1,1
    img = (img.clamp(0, 1) * 255).byte() #0,1 => 0,255
    img = img.permute(1,2,0).cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img,(200,200))
    #archivo = os.path.join("Alumnos", str(i) + ".png")
    #cv2.imwrite(archivo, img)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(hspace=-0.6)
plt.show()