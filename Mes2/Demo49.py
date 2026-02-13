import os, shutil
import pandas as pd

print("Demo 49: Preparando el DataSet de Personas con Pelo Negro y Rubio")
rutaOrigen = "C:/Data/Python/2026_01_IAG/Demos/datasets/img_align_celeba/img_align_celeba/img_align_celeba/"
archivoCsv = r"C:\Data\Python\2026_01_IAG\Demos\datasets\img_align_celeba\list_attr_celeba.csv"
rutaDestino = "C:/Data/Python/2026_01_IAG/Demos/datasets/CelebA/"

df = pd.read_csv(archivoCsv)
print(df)
cpn = 0
cpr = 0
for i in range(len(df)):
    print(f"Procesando Persona {i+1}")
    fila=df.iloc[i]
    archivoOrigen = os.path.join(rutaOrigen, fila["image_id"])
    if(os.path.isfile(archivoOrigen)):
        if(fila["Black_Hair"]==1):
            archivoDestino = os.path.join(rutaDestino + "Black", fila["image_id"])
            try:
                shutil.move(archivoOrigen, archivoDestino)
                cpn=cpn+1
            except Exception as error:
                print("Error: " + str(error))            
        if(fila["Blond_Hair"]==1):
            archivoDestino = os.path.join(rutaDestino + "Blond", fila["image_id"])
            try:
                shutil.move(archivoOrigen, archivoDestino)
                cpr=cpr+1
            except Exception as error:
                print("Error: " + str(error))
print(f"Personas Pelo Negro: {cpn}")
print(f"Personas Pelo Rubio: {cpr}")