# Roger Verzola Peres de Lima 1693271
import sys
import biblioteca
from PyQt5.QtWidgets import QDialog, QApplication
from output import Ui_Form

class AppWindow(QDialog):
    flag = 0
    imagem = 0
    img = 0
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.abrir_arquivo)
        self.ui.pushButton_2.clicked.connect(self.negativo)
        self.ui.pushButton_3.clicked.connect(self.logaritmica)
        self.ui.pushButton_4.clicked.connect(self.potencia)
        self.ui.pushButton_5.clicked.connect(self.histograma)
        self.ui.pushButton_6.clicked.connect(self.contraste)
        self.ui.pushButton_7.clicked.connect(self.equalizado)
        self.ui.pushButton_8.clicked.connect(self.media)
        self.ui.pushButton_9.clicked.connect(self.mediana)
        self.ui.pushButton_10.clicked.connect(self.minimo)
        self.ui.pushButton_11.clicked.connect(self.maximo)
        self.ui.pushButton_12.clicked.connect(self.gaussiano)
        self.ui.pushButton_13.clicked.connect(self.laplaciano)
        self.ui.pushButton_14.clicked.connect(self.set_gray)
        self.ui.pushButton_15.clicked.connect(self.set_hsv)
        self.ui.pushButton_52.clicked.connect(self.limiarizacao_manual)
        self.ui.pushButton_53.clicked.connect(self.limiarizacao_otsu)
        self.ui.pushButton_54.clicked.connect(self.crescimento)
        self.ui.pushButton_55.clicked.connect(self.sobel)
        self.ui.pushButton_56.clicked.connect(self.canny)
        self.ui.pushButton_67.clicked.connect(self.bic)
        self.ui.pushButton_68.clicked.connect(self.abertura)
        self.ui.pushButton_70.clicked.connect(self.cadeia)
        self.ui.pushButton_71.clicked.connect(self.fechamento)
        self.show()

    def abrir_arquivo(self):
        self.imagem = biblioteca.abrir_arquivo()
    def contraste(self):
        biblioteca.contraste(self.imagem, self.flag)
    def equalizado(self):
        biblioteca.equalizado(self.imagem, self.flag)
    def gaussiano(self):
        biblioteca.gaussiano(self.imagem, self.flag)
    def histograma(self):
        biblioteca.histograma(self.imagem, self.flag)
    def laplaciano(self):
        biblioteca.laplaciano(self.imagem, self.flag)
    def logaritmica(self):
        biblioteca.logaritmica(self.imagem, self.flag)
    def maximo(self):
        biblioteca.maximo(self.imagem, self.flag)
    def media(self):
        biblioteca.media(self.imagem, self.flag)
    def mediana(self):
        biblioteca.mediana(self.imagem, self.flag)
    def minimo(self):
        biblioteca.minimo(self.imagem, self.flag)
    def negativo(self):
        biblioteca.negativo(self.imagem, self.flag)
    def potencia(self):
        biblioteca.potencia(self.imagem, self.flag)
    def set_gray(self):
        self.flag = biblioteca.set_gray()
    def set_hsv(self):
        self.flag = biblioteca.set_hsv()
    def limiarizacao_manual(self):
        self.img = biblioteca.limiarizacao_manual(self.imagem)
    def limiarizacao_otsu(self):
        self.img = biblioteca.otsu(self.imagem)
    def crescimento(self):
        self.img = biblioteca.crescimento(self.imagem)
    def sobel(self):
        self.img = biblioteca.bordas_sobel(self.imagem)
    def canny(self):
        self.img = biblioteca.bordas_canny(self.imagem)
    def abertura(self):
        biblioteca.abertura(self.img)
    def fechamento(self):
        biblioteca.fechamento(self.img)
    def bic(self):
        biblioteca.bic(self.imagem)
    def cadeia(self):
        biblioteca.cadeia(self.imagem)


app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())

