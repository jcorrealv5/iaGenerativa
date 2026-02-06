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
        uic.loadUi("dlgImagenDigitos.ui", self)
        #Obtener los Controles de Textos y Botones para Programarlos
        self.txtDigitoInicio = self.findChild(QtWidgets.QSpinBox, "txtDigitoInicio")
        self.txtDigitoFin = self.findChild(QtWidgets.QSpinBox, "txtDigitoFin")
        self.txtNroVeces = self.findChild(QtWidgets.QSpinBox, "txtNroVeces")
        btnProcesarImagen = self.findChild(QtWidgets.QPushButton, "btnProcesarImagen")
        self.lblImagen = self.findChild(QtWidgets.QLabel, "lblImagen")
        self.lblMensaje = self.findChild(QtWidgets.QLabel, "lblMensaje")
        #Programar los eventos clicks de los Botones
        btnProcesarImagen.clicked.connect(self.procesarImagen)
        
    def procesarImagen(self):
        self.n1 = self.txtDigitoInicio.value()
        self.n2 = self.txtDigitoFin.value()
        self.n = self.txtNroVeces.value()
        thread = WorkerModeloVAE(self)
        thread.finalizado.connect(self.mostrarRpta)
        thread.progreso.connect(self.mostrarProgreso)
        thread.start()

    def mostrarProgreso(self, i, imgReconstruida):
        img_final = (imgReconstruida * 255).clip(0, 255).astype(np.uint8)
        #Mostrar la imagen generada en el control Label
        (alto,ancho) = img_final.shape[:2]
        qImg = QImage(img_final.data, ancho, alto, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qImg)
        self.lblImagen.setPixmap(pix)
        self.lblMensaje.setText(f"{i} de {self.n}")

    def mostrarRpta(self, rpta):
        print(rpta)

class WorkerModeloVAE(QThread):
    finalizado = QtCore.pyqtSignal(str)
    progreso = QtCore.pyqtSignal(int, np.ndarray)
    
    def __init__(self, parent):
        super(WorkerModeloVAE, self).__init__(parent)
        self.n1 = parent.n1
        self.n2 = parent.n2
        self.n = parent.n
    
    def run(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        modelo=torch.jit.load('preentrenados/VAE/MNIST/VAE_Digitos_99_130.20439571940105.pt',map_location=device)
        transformacion_data = T.Compose([T.ToTensor()])
        dataset_test = torchvision.datasets.MNIST(root="datasets", train=False, download=True, transform=transformacion_data)
        batchSize = 100
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batchSize, shuffle=True)
        X_test, y_test = next(iter(loader_test))
        x1 = X_test[y_test == self.n1][1].to(device)
        x2 = X_test[y_test == self.n2][1].to(device)
        x1 = x1.view(1, 784).to(device)
        mean1, logvar1 = modelo.encode(x1)
        z1 = modelo.reparameterization(mean1, logvar1)
        x2 = x2.view(1, 784).to(device)
        mean2, logvar2 = modelo.encode(x2)
        z2 = modelo.reparameterization(mean2, logvar2)
        z = torch.stack([z1 + (z2 - z1)*t for t in np.linspace(0, 1, self.n)])
        listaInterpolada = modelo.decode(z)
        arrayInterpolado = listaInterpolada.to('cpu').detach().numpy()
        w = 28
        for i, x_hat in enumerate(arrayInterpolado):
            img = x_hat.reshape(w, w)
            QThread.msleep(100)
            self.progreso.emit(i+1, img)
        rpta = "Se generaron {0} digitos".format(21 * 21)
        self.finalizado.emit(rpta)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    frm = Dialogo()
    frm.show()
    sys.exit(app.exec_())