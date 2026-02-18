import os, shutil
import pandas as pd

print("Demo 58: Preparando Estadistica del DataSet de Celebridades")
rutaOrigen = "C:/Data/Python/2026_01_IAG/Demos/datasets/img_align_celeba/img_align_celeba/img_align_celeba"
archivoCsv = r"C:\Data\Python\2026_01_IAG\Demos\datasets\img_align_celeba\list_attr_celeba.csv"
rutaDestino = "C:/Data/Python/2026_01_IAG/Demos/datasets/Sexo/"

df = pd.read_csv(archivoCsv)
print(df)
cMustache = 0
cSmiling = 0
cChubby = 0
cEyeglasses = 0
for i in range(len(df)):
    print(f"Procesando Persona {i+1}")
    fila=df.iloc[i]
    archivoOrigen = os.path.join(rutaOrigen, fila["image_id"])
    if(os.path.isfile(archivoOrigen)):
        if(fila["Mustache"]==1):
            cMustache=cMustache+1
        if(fila["Smiling"]==1):
            cSmiling=cSmiling+1
        if(fila["Chubby"]==1):
            cChubby=cChubby+1
        if(fila["Eyeglasses"]==1):
            cEyeglasses=cEyeglasses+1
print(f"Personas con Bigote: {cMustache}")
print(f"Personas Sonriendo: {cSmiling}")
print(f"Personas Gorditas: {cChubby}")
print(f"Personas Lentes: {cEyeglasses}")