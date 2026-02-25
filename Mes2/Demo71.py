import sys
sys.path.append("C:/Data/Python/2026_01_IAG/Demos/preentrenados/GAN/StyleGAN3/")
#sys.path.append("C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/")
from urllib.request import urlopen
import torch, dnnlib, os, pickle
import torch_utils
import matplotlib.pyplot as plt

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

print("Demo 71: StyleGAN3")
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
plt.imshow(img)
plt.show()