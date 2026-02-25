import sys
sys.path.append("C:/Data/Python/2026_01_IAG/Demos/preentrenados/GAN/StyleGAN3/")
from urllib.request import urlopen
import torch, dnnlib, os, pickle
import torch_utils
import matplotlib.pyplot as plt
from datetime import datetime

horaInicio = datetime.now()
print("Demo 74: StyleGAN3 del Repositorio de Nvidia con Personas de 1024 x 1024")
archivo = "stylegan3-r-ffhq-1024x1024.pkl"
if(not os.path.isfile(archivo)):
    url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl"
    response = urlopen(url)
    if response.getcode() == 200:    
        with open(archivo, "wb") as f:
            f.write(response.read())
            print(f"Archivo: {archivo} fue creado")
else:
    print(f"El archivo: {archivo} ya existe")
print(f"Cargando Archivo: {archivo}")
with open(archivo, "rb") as f:
    G = pickle.load(f)['G_ema']
print("Creando Ruido")
z = torch.randn([1, G.z_dim])
c = None
print("Generando una Imagen")
imagenesGeneradas = G(z, c)
img = imagenesGeneradas[0]/2+0.5
img = (img.clamp(0, 1) * 255).byte()
img = img.permute(1,2,0).cpu().numpy()
print("Shape Imagen Generada: ", img.shape)

horaFin = datetime.now()
tiempo = (horaFin - horaInicio).total_seconds()
print(f"Tiempo de Procesamiento: {tiempo} seg")

plt.imshow(img)
plt.show()