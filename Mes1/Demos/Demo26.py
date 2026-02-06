import numpy as np
import matplotlib.pyplot as plt
import os, cv2

print("Demo 26: Distribucion Normal o Gaussiana en NumPy con Archivos de Imagen")

archivo = input("Ingresa el archivo de Imagen a representar como Normal: ")
if(os.path.isfile(archivo)):
    imagenGris = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
    print("Shape imagenGris: ", imagenGris.shape)
    plt.imshow(imagenGris, cmap="gray")
    plt.show()
    imagenGrisPlana = imagenGris.ravel()
    print("Shape imagenGrisPlana: ", imagenGrisPlana.shape)
    media = np.mean(imagenGrisPlana)
    ds = np.std(imagenGrisPlana)
    cantidad = imagenGrisPlana.size
    print("Media: ", media)
    print("Desviacion Estandar: ", ds)
    print("Cantidad: ", cantidad)
    normal = np.random.normal(media, ds, cantidad)
    plt.hist(normal, 256)
    plt.axvline(media, color='k', linestyle='dashed', linewidth=2)
    plt.axvline(media - ds, color='k', linestyle='dashed', linewidth=2)
    plt.axvline(media + ds, color='k', linestyle='dashed', linewidth=2)
    plt.show()
else:
    print("No existe el archivo ingresado")