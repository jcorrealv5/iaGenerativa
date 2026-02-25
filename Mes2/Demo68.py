import sys, os, cv2, time
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QDialog, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread
import numpy as np
import torch, torchvision
import torchvision.transforms as T
from PIL import Image
from modGAN import Generador
import albumentations 
from albumentations.pytorch import ToTensorV2

class Dialogo(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        #Cargar la Pantalla o Dialogo en la variable dlg
        uic.loadUi("dlgImagenTransformacion.ui", self)
        #Obtener los Controles de Textos y Botones para Programarlos
        self.txtArchivo = self.findChild(QtWidgets.QLineEdit, "txtArchivo")
        btnAbrirImagen = self.findChild(QtWidgets.QPushButton, "btnAbrirImagen")
        self.cboBigote = self.findChild(QtWidgets.QComboBox, "cboBigote")
        btnProcesarImagen = self.findChild(QtWidgets.QPushButton, "btnProcesarImagen")
        self.lblImagen1 = self.findChild(QtWidgets.QLabel, "lblImagen1")
        self.lblMensaje1 = self.findChild(QtWidgets.QLabel, "lblMensaje1")
        self.lblImagen2 = self.findChild(QtWidgets.QLabel, "lblImagen2")
        self.lblMensaje2 = self.findChild(QtWidgets.QLabel, "lblMensaje2")
        #Programar los eventos clicks de los Botones
        btnAbrirImagen.clicked.connect(self.abrirImagen)
        btnProcesarImagen.clicked.connect(self.procesarImagen)
        #Llenar el Combo de Bigote
        self.bigotes = ["Sin Bigote", "Con Bigote"]
        self.cboBigote.addItems(self.bigotes)

    def abrirImagen(self):
        dlg = QFileDialog()
        dlg.setDirectory("C:/Data/Python/2026_01_IAG/Demos/datasets/TestBigote/No")
        dlg.setNameFilter("Imagenes (*.png *.jpg *.jpeg)");
        dlg.exec()
        self.archivo = dlg.selectedFiles()[0]
        self.txtArchivo.setText(self.archivo)
        pix = QPixmap(self.archivo)
        self.lblImagen1.setPixmap(pix)

    def procesarImagen(self):
        self.bigote = self.cboBigote.currentIndex()
        self.lblMensaje1.setText(f"{self.bigotes[self.bigote]}")
        self.lblMensaje2.setText(f"{self.bigotes[1-self.bigote]}")
        rutaBigote = "C:/Data/Python/2026_01_IAG/Demos/datasets/TestBigote/"
        rutaBigoteOpuesto = rutaBigote + ("Si" if self.bigote==0 else "No")
        archivoBigoteOpuesto = os.path.join(rutaBigoteOpuesto, "Pelado.jpeg" if self.bigote==0 else "Jose_Aljovin.jpeg")
        self.img_Original = np.array(Image.open(self.archivo).convert("RGB"))
        self.img_BigoteOpuesto = np.array(Image.open(archivoBigoteOpuesto).convert("RGB"))
        thread = WorkerModeloGAN(self)
        thread.finalizado.connect(self.mostrarRpta)
        thread.start()

    def mostrarRpta(self, imagenes):
        img = cv2.cvtColor(imagenes[0], cv2.COLOR_BGR2RGB)
        (alto,ancho) = img.shape[:2]
        qImg = QImage(img.data, ancho, alto, ancho*3, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qImg)
        self.lblImagen2.setPixmap(pix)

class WorkerModeloGAN(QThread):
    finalizado = QtCore.pyqtSignal(list)
    
    def __init__(self, parent):
        super(WorkerModeloGAN, self).__init__(parent)
        self.bigote = parent.bigote
        self.img_Original = parent.img_Original
        self.img_BigoteOpuesto = parent.img_BigoteOpuesto

    def convertirTensorToArray(self, imagenesTensor):
        imgs = []
        for i in range(len(imagenesTensor)):
            img = imagenesTensor[i]/2+0.5 #-1,1
            img = (img.clamp(0, 1) * 255).byte() #0,1 => 0,255
            img = img.permute(1,2,0).cpu().numpy()
            imgs.append(img)
        return imgs

    def run(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 1
        transforms = albumentations.Compose(
        [albumentations.Resize(width=256, height=256),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Normalize(mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],max_pixel_value=255),
            ToTensorV2()],
        additional_targets={"image0": "image"})
        #Procesar el Modelo del Bigote
        if(self.bigote==1):
            augmentations = transforms(image=self.img_Original, image0=self.img_BigoteOpuesto)
        else:
            augmentations = transforms(image=self.img_BigoteOpuesto, image0=self.img_Original)
        imgSinBigote = augmentations["image0"]
        imgConBigote = augmentations["image"]
        dstBigote = [(imgSinBigote, imgConBigote)]
        loaderBigote=torch.utils.data.DataLoader(dstBigote,batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        gen_A = Generador(img_channels=3, num_residuals=9).to(device)
        gen_B = Generador(img_channels=3, num_residuals=9).to(device)
        gen_A.load_state_dict(torch.load("preentrenados/GAN/Bigote/gen_sinbigote_6.pth"))
        gen_B.load_state_dict(torch.load("preentrenados/GAN/Bigote/gen_conbigote_6.pth"))
        for sinbigote,conbigote in loaderBigote:
            if(self.bigote==0):
                fakeBigote=gen_B(sinbigote.to(device)).squeeze(0)
            else:
                fakeBigote=gen_A(conbigote.to(device)).squeeze(0)        
        imagenesTensor = [fakeBigote]
        imagenesArray = self.convertirTensorToArray(imagenesTensor)
        self.finalizado.emit(imagenesArray)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    frm = Dialogo()
    frm.show()
    sys.exit(app.exec_())