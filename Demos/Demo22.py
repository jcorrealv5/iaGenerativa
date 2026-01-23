import sys, os, cv2
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QDialog, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from modVAE import VAE

class Dialogo(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        #Cargar la Pantalla o Dialogo en la variable dlg
        uic.loadUi("dlgTransicionImagen.ui", self)
        #Obtener los Controles de Textos y Botones para Programarlos
        self.txtArchivo = self.findChild(QtWidgets.QLineEdit, "txtArchivo")
        btnAbrirArchivo = self.findChild(QtWidgets.QPushButton, "btnAbrirArchivo")
        btnNuevaImagen = self.findChild(QtWidgets.QPushButton, "btnNuevaImagen")
        btnProcesarImagen = self.findChild(QtWidgets.QPushButton, "btnProcesarImagen")
        self.lblCara1 = self.findChild(QtWidgets.QLabel, "lblCara1")
        self.lblCara2 = self.findChild(QtWidgets.QLabel, "lblCara2")
        self.lblCara3 = self.findChild(QtWidgets.QLabel, "lblCara3")
        self.lblMensaje = self.findChild(QtWidgets.QLabel, "lblMensaje")
        self.cboCara = self.findChild(QtWidgets.QComboBox, "cboCara")
        self.txtNroVeces = self.findChild(QtWidgets.QSpinBox, "txtNroVeces")
        #Programar los eventos clicks de los Botones
        btnAbrirArchivo.clicked.connect(self.abrirArchivo)
        btnNuevaImagen.clicked.connect(self.nuevaImagen)
        btnProcesarImagen.clicked.connect(self.procesarImagen)
        #Llenar el Combo con 2 opciones
        self.cboCara.addItem("Cara 1")
        self.cboCara.addItem("Cara 2")

    def fileToTensor(self, archivo):
        transform = T.Compose([T.ToTensor(),T.Resize(256)])
        img = Image.open(archivo)
        img_tensor = transform(img)
        return img_tensor
        
    def abrirArchivo(self):
        dlg= QFileDialog()
        dlg.exec()
        self.archivo = dlg.selectedFiles()[0]
        self.txtArchivo.setText(self.archivo)
        pix = QPixmap(self.archivo)
        if(self.cboCara.currentIndex()==0):
            self.lblCara1.setPixmap(pix)
            self.tensorCara1 = self.fileToTensor(self.archivo)
        else:
            self.lblCara2.setPixmap(pix)
            self.tensorCara2 = self.fileToTensor(self.archivo)

    def nuevaImagen(self):
        self.txtArchivo.setText("")
        self.lblCara1.setPixmap(QPixmap())
        self.lblCara2.setPixmap(QPixmap())
        self.lblCara3.setPixmap(QPixmap())
        self.lblMensaje.setText("")

    def procesarImagen(self):
        self.nroVeces = self.txtNroVeces.value()
        thread = WorkerModeloVAE(self)
        thread.finalizado.connect(self.mostrarRpta)
        thread.progreso.connect(self.mostrarProgreso)
        thread.start()

    def mostrarProgreso(self, item, imgReconstruida):
        self.lblMensaje.setText(f"{item} - {self.nroVeces}")
        #Convertir el Tensor generado en una imagen o array de NumPy RGB
        img_tensor = imgReconstruida.reshape((3, 256, 256))
        img_array = np.transpose(img_tensor.detach().cpu().numpy(), (1, 2, 0))
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)               
        img_final = (img_rgb * 255).clip(0, 255).astype(np.uint8)
        #Mostrar la imagen generada en el control Label
        (alto,ancho) = img_final.shape[:2]
        qImg = QImage(img_final.data, ancho, alto, 3 * ancho, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qImg)
        self.lblCara3.setPixmap(pix)

    def mostrarRpta(self, rpta):
        print(rpta)

class WorkerModeloVAE(QThread):
    finalizado = QtCore.pyqtSignal(str)
    progreso = QtCore.pyqtSignal(int, torch.Tensor)
    
    def __init__(self, parent):
        super(WorkerModeloVAE, self).__init__(parent)
        self.tensorCara1 = parent.tensorCara1
        self.tensorCara2 = parent.tensorCara2
        self.nroVeces = parent.nroVeces
    
    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        modelo=VAE().to(device)
        modelo.eval()
        modelo.load_state_dict(torch.load('preentrenados/VAE/Lentes/VAE_Lentes_487_512.5182352437602.pt', map_location=device))
        cara1 = self.tensorCara1.unsqueeze(0)
        cara2 = self.tensorCara2.unsqueeze(0)
        _,_,code1=modelo.encoder(cara1)
        _,_,code2=modelo.encoder(cara2)
        c = 0
        for w in np.linspace(0,1,self.nroVeces):
            c=c+1
            z=w*code2+(1-w)*code1
            out=modelo.decoder(z)
            QThread.msleep(100)
            self.progreso.emit(c, out)
        self.finalizado.emit("Finalizo")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    frm = Dialogo()
    frm.show()
    sys.exit(app.exec_())