import os, shutil
import pandas as pd

print("Demo 52: Preparando el DataSet de Personas por Sexo")

rutaOrigen = "../../datasets/CelebA"
archivoCsv = r"../../datasets/img_align_celeba/list_attr_celeba.csv"
rutaDestino = "../../datasets/Sexo/"


df = pd.read_csv(archivoCsv)
print(df)
ch = 0
cm = 0
t=5000  # Número de imágenes por cada sexo a mover
for i in range(len(df)):
    print(f"Procesando Persona {i+1}")
    fila=df.iloc[i]
    archivoOrigen = os.path.join(rutaOrigen, fila["image_id"])
    if(os.path.isfile(archivoOrigen)):
        if(fila["Male"]==1):
            archivoDestino = os.path.join(rutaDestino + "Masculino", fila["image_id"])
            try:
                shutil.move(archivoOrigen, archivoDestino)
                ch=ch+1
                if(ch==t):
                    break
            except Exception as error:
                print("Error: " + str(error))        
        if(fila["Male"]==-1):
            archivoDestino = os.path.join(rutaDestino + "Femenino", fila["image_id"])
            try:
                shutil.move(archivoOrigen, archivoDestino)
                cm=cm+1
            except Exception as error:
                print("Error: " + str(error))
print(f"Personas Sexo Masculino: {ch}")
print(f"Personas Sexo Femenino: {cm}")