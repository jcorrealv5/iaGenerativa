import pandas as pd
import os, shutil

print("Demo 65: Crear un DataSet de Bigotes para Hombres usando CelebA")
rutaOrigen = "C:/Data/Python/2026_01_IAG/Demos/datasets/img_align_celeba/img_align_celeba/img_align_celeba/"
rutaDestino = "C:/Data/Python/2026_01_IAG/Demos/datasets/Bigotes/"
archivo = "list_attr_celeba.csv"
df = pd.read_csv(archivo)
nRegistros = len(df)
cSi = 0
cNo = 0
totalHombres = 0
for i in range(nRegistros):
    fila = df.iloc[i]
    if(fila["Male"]==1):        
        nombre = fila["image_id"]
        archivoOrigen = os.path.join(rutaOrigen, nombre)
        if(os.path.isfile(archivoOrigen)):
            totalHombres += 1
            print(f"Procesando Archivo fila: {i}")
            if(fila["Mustache"]==1):
                try:
                    archivoDestino = os.path.join(rutaDestino + "Si", nombre) 
                    shutil.move(archivoOrigen, archivoDestino)
                    cSi += 1
                except Exception as error:
                    print(f"Error Si: {str(error)}")
            else:
                if(fila["No_Beard"]==1):
                    try:
                        archivoDestino = os.path.join(rutaDestino + "No", nombre)
                        shutil.move(archivoOrigen, archivoDestino)
                        cNo += 1
                    except Exception as error:
                        print(f"Error No: {str(error)}")
        else:
            print(f"Archivo No existe en fila: {i}")
print(f"Imagenes con Bigotes: {cSi}")
print(f"Imagenes sin Bigotes: {cNo}")
print(f"Total de Hombres: {totalHombres}")
print(f"Total del DataFrame: {nRegistros}")