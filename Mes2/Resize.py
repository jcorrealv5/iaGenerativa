import os, cv2

rutaOrigen = "C:/Data/Python/2026_01_IAG/Demos/datasets/Temp/"
rutaDestino = "C:/Data/Python/2026_01_IAG/Demos/datasets/TestSonrisa/"
archivos = os.listdir(rutaOrigen)
for archivo in archivos:    
    print(archivo)
    imagen = cv2.imread(os.path.join(rutaOrigen, archivo))
    imagen = cv2.resize(imagen, (256,256))
    cv2.imwrite(os.path.join(rutaDestino, archivo), imagen)