import os, shutil
import pandas as pd

print("Demo 59: Preparando el DataSet de Celebreridades Sonriendo")
rutaOrigen = "C:/Data/Python/2026_01_IAG/Demos/datasets/img_align_celeba/img_align_celeba/img_align_celeba"
archivoCsv = r"C:\Data\Python\2026_01_IAG\Demos\datasets\img_align_celeba\list_attr_celeba.csv"
rutaDestino = "C:/Data/Python/2026_01_IAG/Demos/datasets/Sonriendo/"

df = pd.read_csv(archivoCsv)
print(df)
cSi = 0
cNo = 0
for i in range(len(df)):
    print(f"Procesando Persona {i+1}")
    fila=df.iloc[i]
    archivoOrigen = os.path.join(rutaOrigen, fila["image_id"])
    if(os.path.isfile(archivoOrigen)):
        if(fila["Smiling"]==1):
            archivoDestino = os.path.join(rutaDestino + "Si", fila["image_id"])
            try:
                shutil.move(archivoOrigen, archivoDestino)
                cSi=cSi+1
            except Exception as error:
                print("Error: " + str(error))        
        else:
            archivoDestino = os.path.join(rutaDestino + "No", fila["image_id"])
            try:
                shutil.move(archivoOrigen, archivoDestino)
                cNo=cNo+1
            except Exception as error:
                print("Error: " + str(error))
print(f"Personas Sonriendo: {cSi}")
print(f"Personas No Sonriendo: {cNo}")