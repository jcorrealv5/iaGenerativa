import torch, torchvision, os, cv2

print("Demo 73: Usando Torch-Hub GAN Zoo Models - PGAN para generar Archivos con Caras en Disco\n")
directorio = input("Ingresa el Directorio donde deseas generar los Archivos: ")
if(os.path.isdir(directorio)):
    nMuestras = int(input("\nNumero de Muestras o Archivos a Generar: "))
    use_gpu = True if torch.cuda.is_available() else False
    print("GPU: ", use_gpu)
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub','PGAN', model_name='celebAHQ-512', pretrained=True, useGPU=use_gpu)
    print("\nGenerando Imagenes a Disco...")
    nBloques = int(nMuestras / 10)
    if(nMuestras % 10>0):
        nBloques += 1
    c = 0
    for i in range(nBloques):
        nImagenes = 10
        if((i==nBloques-1) and (nMuestras%10>0)):
            nImagenes = nMuestras%10
        print(f"i: {i}, nImagenes: {nImagenes}")
        ruido, _ = model.buildNoiseData(nImagenes)
        with torch.no_grad():
            imagenesGeneradas = model.test(ruido)
        for j in range(nImagenes):
            print(f"Imagen: {c+1}")
            img = imagenesGeneradas[j]/2+0.5
            img = (img.clamp(0, 1) * 255).byte()
            img = img.permute(1,2,0).cpu().numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            archivo = os.path.join(directorio, str(c+1) + ".png")
            cv2.imwrite(archivo, img)
            c += 1
    print(f"Se crearon {nMuestras} archivos en disco")
else:
    print("Directorio Destino No existe")
