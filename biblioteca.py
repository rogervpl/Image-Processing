# Roger Verzola Peres de Lima 1693271
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import arff

def abrir_arquivo():
    tk.Tk().withdraw()  # ocultar a janela raiz
    arq = filedialog.askopenfilename()
    img = cv.imread(arq)
    cv.imshow("Imagem Colorida", img)
    return img

def original(img, flag):
    if flag == 0:
        cinza = entrada(img, flag)
        cv.imshow("Imagem Cinza", cinza)
    else:
        cv.imshow("Imagem Original", img)

def negativo(img, flag):
    cinza = entrada(img, flag)
    linhas, colunas = cinza.shape
    negativo = cinza.copy()
    for x in range(linhas):
        for y in range(colunas):
            negativo[x, y] = 255 - cinza[x, y]
    cv.destroyAllWindows()
    said = saida(negativo, flag)
    original(img, flag)
    cv.imshow("Imagem Negativo", said)
    return said


def logaritmica(img, flag):
    cinza = entrada(img, flag)
    linhas, colunas = cinza.shape
    logaritmica = cinza.copy()
    c = 255 / (math.log(1 + img.max(), 2))
    
    for x in range(linhas):
	    for y in range(colunas):
		    logaritmica[x, y] = c * (math.log(1 + logaritmica[x, y], 2))
    cv.destroyAllWindows()
    said = saida(logaritmica, flag)
    original(img, flag)
    cv.imshow("Transformacao Logaritmica", said)
    return said

def potencia(img, flag):
    cinza = entrada(img, flag)

    linhas, colunas = cinza.shape
    potencia = cinza.copy()
    print("entre com o c")
    c = float(input())
    print("entre com o gama")
    gama = float(input())

    for x in range(linhas):
        for y in range(colunas):
            potencia[x, y] = 255 * (math.pow(potencia[x, y] / 255, 1 / gama))
    cv.destroyAllWindows()
    said = saida(potencia, flag)
    original(img, flag)
    cv.imshow("Transformacao Potencia", said)
    return said


def calc_histograma(img, flag):
    f = entrada(img, flag)
    cinza = f.copy()
    h = [0]*256
    linhas, colunas = cinza.shape
    for x in range(linhas):
        for y in range(colunas):
            h[cinza[x,y]]+=1
    return h


def histograma(img, flag):
    h = calc_histograma(img, flag)
    plt.bar(np.arange(len(h)),h)
    plt.title("Histograma da Imagem")
    plt.xlabel("Valor de Intensidade")
    plt.ylabel("Frequencia dos Niveis de Intensidade")
    plt.show()

def contraste(img, flag):
    imagem = entrada(img, flag)
    f = imagem.copy()
    linhas, colunas = f.shape
    g = f.copy()
    gmax = 255
    gmin = 0
    for x in range(linhas):
        for y in range(colunas):
            g[x, y] = ((gmax - gmin) / (f.max() - f.min())) * (f[x, y] - f.min()) + gmin
    cv.destroyAllWindows()
    said = saida(g, flag)
    original(img, flag)
    cv.imshow("Ajuste de contraste", said)
    return said

def equalizado(img, flag):
    f = entrada(img, flag)
    histograma = calc_histograma(img, flag)
    linhas, colunas = f.shape
    total = linhas*colunas
    normalizado = list(map(lambda n:n/total, histograma))
    soma = 0
    acumulado = []
    equalizada = f.copy()
    for i in normalizado:
        soma = soma + i
        acumulado.append(soma)
    s = list(map(lambda n: round(255*n), acumulado))
    for x in range(linhas):
        for y in range(colunas):
            equalizada[x,y] = s[f[x,y]]
    cv.destroyAllWindows()
    said = saida(equalizada, flag)
    original(img, flag)
    cv.imshow("Histograma Equalizado", said)
    return said


def convolucao(img, mascara):
    f = img.copy()
    linhas, colunas = f.shape
    reflect = cv.copyMakeBorder(f,1,1,1,1, cv.BORDER_REFLECT)
    g = np.zeros_like(reflect)
    for x in range(linhas):
        for y in range(colunas):
            g[x,y]=(mascara*reflect[x:x+3,y:y+3]).sum()
    g=g[:linhas,:colunas]
    return g

def media(img, flag):
    imagem = entrada(img, flag)
    mascara = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0
    g = convolucao(imagem, mascara)
    cv.destroyAllWindows()
    said = saida(g, flag)
    original(img, flag)
    cv.imshow("Filtro Media", said)
    return said

def mediana(img, flag):
    imagem = entrada(img, flag)
    f = imagem.copy()
    linhas, colunas = f.shape
    g = cv.copyMakeBorder(f, 1, 1, 1, 1, cv.BORDER_REFLECT)
    mascara = [(0, 0)] * 9
    for x in range(linhas):
        for y in range(colunas):
            mascara[0] = g[x - 1][y - 1]
            mascara[1] = g[x - 1][y]
            mascara[2] = g[x - 1][y + 1]
            mascara[3] = g[x][y - 1]
            mascara[4] = g[x][y]
            mascara[5] = g[x][y + 1]
            mascara[6] = g[x + 1][y - 1]
            mascara[7] = g[x + 1][y]
            mascara[8] = g[x + 1][y + 1]
            mascara.sort()
            f[x, y] = mascara[4]
    cv.destroyAllWindows()
    said = saida(f, flag)
    original(img, flag)
    cv.imshow("Filtro Mediana", said)
    return said
    

def minimo(img, flag):
    imagem = entrada(img, flag)
    f = imagem.copy()

    linhas, colunas = f.shape
    g = cv.copyMakeBorder(f, 1, 1, 1, 1, cv.BORDER_REFLECT)

    mascara = [(0, 0)] * 9

    for x in range(linhas):
        for y in range(colunas):
            mascara[0] = g[x - 1][y - 1]
            mascara[1] = g[x - 1][y]
            mascara[2] = g[x - 1][y + 1]
            mascara[3] = g[x][y - 1]
            mascara[4] = g[x][y]
            mascara[5] = g[x][y + 1]
            mascara[6] = g[x + 1][y - 1]
            mascara[7] = g[x + 1][y]
            mascara[8] = g[x + 1][y + 1]
            mascara.sort()
            f[x, y] = min(mascara)
    cv.destroyAllWindows()
    said = saida(f, flag)
    original(img, flag)
    cv.imshow("Filtro Minimo", said)
    return said
   

def maximo(img, flag):
    imagem = entrada(img, flag)
    f = imagem.copy()

    linhas, colunas = f.shape
    g = cv.copyMakeBorder(f, 1, 1, 1, 1, cv.BORDER_REFLECT)
    mascara = [(0, 0)] * 9

    for x in range(linhas):
        for y in range(colunas):
            mascara[0] = g[x - 1][y - 1]
            mascara[1] = g[x - 1][y]
            mascara[2] = g[x - 1][y + 1]
            mascara[3] = g[x][y - 1]
            mascara[4] = g[x][y]
            mascara[5] = g[x][y + 1]
            mascara[6] = g[x + 1][y - 1]
            mascara[7] = g[x + 1][y]
            mascara[8] = g[x + 1][y + 1]
            mascara.sort()
            f[x, y] = max(mascara)
    cv.destroyAllWindows()
    said = saida(f, flag)
    original(img, flag)
    cv.imshow("Filtro Maximo", said)
    return said


def convolucao5(img,mascara):
    f = img.copy()
    linhas, colunas = f.shape
    reflect = cv.copyMakeBorder(f,2,2,2,2, cv.BORDER_REFLECT)
    g = np.zeros_like(reflect)
    for x in range(linhas):
        for y in range(colunas):
            g[x,y]=(mascara*reflect[x:x+5,y:y+5]).sum()
    g=g[:linhas,:colunas]
    return g

def gaussiano(img, flag):
    mascara_gaussiana = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]]) / 273.0
    imagem = entrada(img, flag)
    f = convolucao5(imagem, mascara_gaussiana)
    cv.destroyAllWindows()
    said = saida(imagem,flag)
    original(img, flag)
    cv.imshow("Filtro Gaussiano", said)
    return said

def laplaciano(img, flag):
    imagem = entrada(img, flag)
    imagem = imagem.astype(np.float32)
    mascara = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    filtrada = convolucao(imagem, mascara)

    f = imagem + filtrada
    f = np.clip(f, 0, 255)
    f = f.astype(np.uint8)

    filtrada = np.clip(filtrada, 0, 255)
    filtrada = filtrada.astype(np.uint8)
    cv.destroyAllWindows()
    said = saida(f, flag)
    filtrada = saida(filtrada, flag)
    original(img, flag)
    cv.imshow("Filtro Laplaciano", said)
    return said

def set_gray():
    return 0

def set_hsv():
    return 1

def entrada (img, flag):
    if flag == 0:
        entrada = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        global hsv
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        entrada = v.copy()
    return entrada

def saida(img, flag):
    if flag == 0:
        imagem = img.copy()
        imagem = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        return imagem
    else:
        global hsv
        hsv[:, :, 2] = img.copy()
        imagem = img.copy()
        imagem = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        return imagem	

def limiarizacao_manual(img):
    flag = 0
    limiar = input("entre com limiar: ")
    limiar = int(limiar)
    imagem = entrada(img, flag)
    linhas, colunas = imagem.shape
    for x in range(linhas):
        for y in range(colunas):
            if imagem[x,y] >= limiar:
                imagem[x,y] = 255
            else:
                imagem[x,y] = 0
    cv.destroyAllWindows()
    said = saida(imagem, flag)
    original(img, flag)
    cv.imshow("Limiarizacao", said)
    return said
    

def otsu(img):
    flag = 0
    imagem = entrada(img, flag)
    lim, res = cv.threshold(imagem, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.destroyAllWindows()
    said = saida(res, flag)
    original(img, flag)
    cv.imshow("Limiarizacao Otsu", res)
    return said

def crescimento(img):
    flag = 0
    imagem = entrada(img, flag)
    x0 = input("entre com x0: ")
    y0 = input("entre com y0: ")
    deg = input("entre com o grau: ")

    x0 = int(x0)
    y0 = int(y0)
    deg = int(deg)
    
    linhas, colunas = imagem.shape

    for x in range(linhas):
        for y in range(colunas):
            sub = int(imagem[x,y]) - int(imagem[x0,y0])
            if sub >= -deg and sub<= deg:
                imagem[x,y] = 255
            else:
                imagem[x,y] = 0
    cv.destroyAllWindows()
    said = saida(imagem, flag)
    original(img, flag)
    cv.imshow("Crescimento de Regioes", said)
    return said

def bordas_sobel(img):
    flag = 0
    vert = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    hori = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    imagem = entrada(img, flag)
    imagem = imagem.astype(np.float32)
    out1 = convolucao(imagem, vert)
    out2 = convolucao(imagem, hori)
    said = out1 + out2
    said = np.clip(said, 0, 255)
    said = said.astype(np.uint8)
    said = saida(said, flag)
    said = otsu(said)
    cv.destroyAllWindows()
    original(img, flag)
    cv.imshow("Imagem Sobel", said)
    return said

def bordas_canny(img):
    flag = 0
    imagem = entrada(img, flag)
    lim_inf = input("limite inferior: ")
    razao = input("razao: ")

    lim_inf = int(lim_inf)
    razao = int(razao)
    bord = cv.Canny(img, lim_inf, lim_inf*razao, 3)
    said = saida(bord, flag)
    cv.destroyAllWindows()
    original(img, flag)
    cv.imshow("Imagem Canny", said)
    return said

def abertura(img):
    flag = 0
    said = minimo(img, flag)
    said = maximo(said, flag)
    cv.destroyAllWindows()
    original(img, flag)
    cv.imshow("Imagem Abertura", said)
    return said

def fechamento(img):
    flag = 0
    said = maximo(img, flag)
    said = minimo(said, flag)
    cv.destroyAllWindows()
    original(img, flag)
    cv.imshow("Imagem Fechamento", said)
    return said

def cadeia(img):
    flag = 0
    imagem = entrada(img, flag)

    aux, contornos, aux1 = cv.findContours(imagem, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    cadeia = []
    #Dica de usar chaves em https://www.geeksforgeeks.org/chain-code-for-2d-line/
    lista = [3, 4, 5, 2, -1, 6, 1, 0, 7]
    tam = len(contornos[0])
    for i in range(tam):
        a = contornos[0][i][0]
        if i < tam - 1:
            b = contornos[0][i + 1][0]

            x = b[0] - a[0]
            y = b[1] - a[1]
            ch = 3*y + x + 4

            cadeia.append(lista[ch])
            
        else:
            i = contornos[0][0][0]

            x = i[0] - a[0]
            y = i[1] - a[1]
            ch = 3*y + x + 4

            cadeia.append(lista[ch])        
    atrib = []
    for i in range(tam):
        item = ('c{}'.format(i), 'INTEGER')
        atrib.append(item)
    obj = {
        'relation': 'chain code',
        'attributes': atrib,
        'data': [cadeia],
    }
    f = open("chaincode.arff", "w", encoding="utf-8")
    arff.dump(obj, f)
    f.close()

def bic(img):
    imagem = img.copy()
    ima = cv.copyMakeBorder(imagem, 1, 1, 1, 1, cv.BORDER_REFLECT)
    masc = imagem.copy()
    for x in range(imagem.shape[0]):
        for y in range(imagem.shape[1]):
            if (ima[x+1, y, 0]==imagem[x, y, 0]==ima[x, y+1, 0]==ima[x+2, y+1, 0]==ima[x+1, y+2, 0] and
            ima[x+1, y, 1]==imagem[x, y, 1]==ima[x, y+1, 1]==ima[x+2, y+1, 1]==ima[x+1, y+2, 1] and
            ima[x+1, y, 2]==imagem[x, y, 2]==ima[x, y+1, 2]==ima[x+2, y+1, 2]==ima[x + 1, y + 2, 2]):
                masc[x, y, 0] = 255
                masc[x, y, 1] = 255
                masc[x, y, 2] = 255
            else:
                masc[x, y, 0] = 0
                masc[x, y, 1] = 0
                masc[x, y, 2] = 0
    hist = cv.calcHist([imagem], [0,1,2], masc[:,:,0], [4,4,4], [0,256,0,256,0,256])
    hist = hist.flatten()
    masc1 = np.ones_like(imagem)*255
    masc1 = masc1 - masc
    hist1 = cv.calcHist([imagem], [0,1,2], masc1[:,:,0], [4,4,4], [0,256,0,256,0,256])
    hist1 = hist1.flatten()
    histf = np.concatenate([hist1, hist])
    histf = np.uint32(histf)
    atrib = []
    for i in range(128):
        item = ('h{}'.format(i), 'INTEGER')
        atrib.append(item)
    obj = {
        'relation': 'bic',
        'attributes': atrib,
        'data': [histf],
    }
    f = open("bic.arff", "w", encoding="utf-8")
    arff.dump(obj, f)
    f.close()

	
	
	
