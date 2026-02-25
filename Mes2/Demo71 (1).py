import torch, torchvision, os, cv2

print("Demo 71: Usando Torch-Hub GAN Zoo Models - DCGAN para generar Archivos de Ropa en Disco\n")
directorio = input("Ingresa el Directorio donde deseas generar los Archivos: ")
if(os.path.isdir(directorio)):
    nMuestras = int(input("\nNumero de Muestras o Archivos a Generar: "))
    use_gpu = True if torch.cuda.is_available() else False
    print("GPU: ", use_gpu)
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub','DCGAN', pretrained=True, useGPU=use_gpu)
    ruido, _ = model.buildNoiseData(nMuestras)
    with torch.no_grad():
        imagenesGeneradas = model.test(ruido)
    print("\nGenerando Imagenes a Disco...")
    for i in range(nMuestras):
        print(f"Imagen: {i+1}")
        img = imagenesGeneradas[i]/2+0.5
        img = (img.clamp(0, 1) * 255).byte()
        img = img.permute(1,2,0).cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        archivo = os.path.join(directorio, str(i+1) + ".png")
        cv2.imwrite(archivo, img)
    print(f"Se crearon {nMuestras} archivos en disco")
else:
    print("Directorio Destino No existe")
