import sys, os, cv2
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QDialog, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread
import numpy as np
import torch, torchvision
import torchvision.transforms as T

class Dialogo(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        #Cargar la Pantalla o Dialogo en la variable dlg
        uic.loadUi("dlgImagen.ui", self)
        #Obtener los Controles de Textos y Botones para Programarlos
        btnProcesarImagen = self.findChild(QtWidgets.QPushButton, "btnProcesarImagen")
        self.lblImagen = self.findChild(QtWidgets.QLabel, "lblImagen")
        self.lblMensaje = self.findChild(QtWidgets.QLabel, "lblMensaje")
        #Programar los eventos clicks de los Botones
        btnProcesarImagen.clicked.connect(self.procesarImagen)
        
    def procesarImagen(self):
        thread = WorkerModeloVAE(self)
        thread.finalizado.connect(self.mostrarRpta)
        thread.progreso.connect(self.mostrarProgreso)
        thread.start()

    def mostrarProgreso(self, n, imgReconstruida):
        #Convertir el Tensor generado en una imagen o array de NumPy RGB
        img_array = np.transpose(imgReconstruida.numpy(),(1,2,0)) 
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_final = (img_rgb * 255).clip(0, 255).astype(np.uint8)
        #Mostrar la imagen generada en el control Label
        (alto,ancho) = img_final.shape[:2]
        qImg = QImage(img_final.data, ancho, alto, ancho*3, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qImg)
        self.lblImagen.setPixmap(pix)
        total = 30*30
        self.lblMensaje.setText(f"{n} de {total}")

    def mostrarRpta(self, rpta):
        print(rpta)

class WorkerModeloVAE(QThread):
    finalizado = QtCore.pyqtSignal(str)
    progreso = QtCore.pyqtSignal(int, torch.Tensor)
    
    def __init__(self, parent):
        super(WorkerModeloVAE, self).__init__(parent)
    
    def run(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        modelo=torch.jit.load('preentrenados/VAE/Caras/VAE_Caras_998_79.27378463745117.pt',map_location=device)
        transform = T.Compose([T.ToTensor(), T.Resize(100)])
        dataset_test = torchvision.datasets.ImageFolder(root="datasets/Voluntarios", transform=transform)
        batch_size=60
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,shuffle=True)
        n=30
        y = 2.5
        c = 0
        for i in range(n):
            x = -2.5            
            for j in range(n):
                c=c+1
                z_sample = torch.tensor([[x, y]], dtype=torch.float).to(device)
                x_decoded = modelo.decode(z_sample)
                caraGenerada = x_decoded.detach().cpu().reshape(3, 100, 100)
                QThread.msleep(50)
                self.progreso.emit(c,caraGenerada)
                x += (5/(n-1))
            y -= (5/(n-1))
        rpta = "Se generaron {0} caras".format(n * n)
        self.finalizado.emit(rpta)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    frm = Dialogo()
    frm.show()
    sys.exit(app.exec_())