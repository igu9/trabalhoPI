# PRETO = 0
# BRANCO = 255
import os, time, random

import cv2 #opencv

from matplotlib import pyplot as plt
import numpy as np

# janela gráfica
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox

from PIL import ImageTk, Image  # pip install Pillow

from mnist import MNIST


# variavel global que vai guardar a img selecionada
imgPath = None
img = None
imgcpy = None


janela = tk.Tk()
largura = 800
altura = 600

def redimensiona_imagem(imagem):
    if imagem.width > 504:
        if imagem.height > 504:
            imagem = imagem.resize((504, 504))
        else: imagem = imagem.resize((504, imagem.height))
    else:
        if imagem.height > 504:
            imagem = imagem.resize((imagem.width, 504))
    print("redimensiona_imagem")
    print("imagem.width = " + str(imagem.width))
    print("imagem.height = " + str(imagem.height))

    # imagem fica de ponta-cabeça, girar ela p/ orientacao normal
    imagem = imagem.rotate(180)
    return imagem


def trata_img():
    print("trata_img")
    
    
    # transforma imagem para numpy array com 3 dimensoes (width, height, 3)
    imgArray = np.array(img.copy(), dtype="int")
    print(imgArray.shape)
    print(str(type(imgArray)))
    
    # y, x, z = imgArray.shape
    # print("y = " + str(y) + "  x = " + str(x))
    # imgArray = imgArray.reshape((y, x))
    # print(imgArray.shape)
    # print(str(type(imgArray)))

    # passa a img p/ tons de cinza e a redimension (se necessario)
    img_NC = Image.open(imgPath).convert('L')
    if img_NC.width > 560 or img_NC.height > 560: 
        img_NC = redimensiona_imagem(img_NC)
    print(img_NC)

    global imgcpy
    imgcpy = ImageTk.PhotoImage(img_NC.copy())
    
    # print("sleep - NC")
    # time.sleep(2)

    lblImg = tk.Label(image = imgcpy)

    lblImg.place(x = 15, y = 50)

    # transforma a imagem em array numpy (ndarray)
    imgArray = np.array(img_NC.copy(), dtype="int")
    print(imgArray.shape)
    print(str(type(imgArray)))

    toPNG(imgArray, "img-importada")
    src = cv2.imread('C:/Users/Igor/Desktop/Dropbox/6osem/pi/tp/pi-trabalho/img-importada.png')
    
    # filtro gaussiano
    img_filtroGauss = cv2.GaussianBlur(src, (5, 5), 0) # kernel 3


    # printar mais de uma img na msm tela
    print("subplots")
    linhas = 1
    colunas = 2
    tela = plt.figure(figsize=(10, 5))

    tela.add_subplot(linhas, colunas, 1) # 1a imagem
    plt.imshow(imgArray, cmap="gray")
    plt.axis("off")
    plt.title("img importada - NC")
    

    tela.add_subplot(linhas, colunas, 2) # 2a imagem
    plt.imshow(img_filtroGauss, cmap="gray")
    plt.axis("off")
    plt.title("img importada - gaussiano")

    # plotar img 3 -> imagem 1 limiarizada
    # plotar img 4 -> imagem 2 limiarizada

    plt.show()

def carregar_imagem():
    global imgPath
    imgPath = fd.askopenfilename()
    print(imgPath)
    
    global img 
    img = Image.open(imgPath)
    
    print("largura = " + str(img.width))
    print("altura = " + str(img.height))

    if img.width > 560 or img.height > 560: 
        img = redimensiona_imagem(img)
    print("redimensionou img")
    print("img.width = " + str(img.width))
    print("img.height = " + str(img.height))
    
    global imgcpy
    imgcpy = ImageTk.PhotoImage(img.copy())

    # print("em carregar_imagem: " + str(type(imgcpy)))

    lblImg = tk.Label(image = imgcpy)

    lblImg.place(x = 15, y = 50)

    # janela.mainloop()
    # trata_img()

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        janela.destroy()


def inicializa_janela():

    # parametros da interface grafica
    janela.geometry(str(largura) + "x" + str(altura))
    janela.columnconfigure(0, weight=1)
    janela.rowconfigure(0, weight=1)
    janela.title("OCR")
    
    # janela.protocol("WM_DELETE_WINDOW", on_closing) # confirmacao ao fechar a janela

    # botoes e rotulos - declaracao
    btnCarregarImg = tk.Button(janela, text = "Carregar imagem", bg = "#7E7E7E", fg = "#ffffff", font = ("Calibri", 10), command = lambda : carregar_imagem())
    btnTratarImg = tk.Button(janela, text = "Tratar imagem", bg = "#7E7E7E", fg = "#ffffff", font = ("Calibri", 10), command = lambda : trata_img())

    # botoes e rotulos - posicao
    btnCarregarImg.place(x = 15, y = 20)
    btnTratarImg.place(x = 135, y = 20)

# "printa" a imagem no terminal
def printa_imagem(img):
    for linha in img:
        for px in linha:
            print(px, end = '\t')
        print()

# retorna um vetor que representa a projecao horizontal da imagem
def projHorizontal(img):
  
    # Convert black spots to ones
    img[img == 0]   = 1
    # Convert white spots to zeros
    img[img == 255] = 0
  
    projecao_horizontal = np.sum(img, axis = 1) 
  
    return projecao_horizontal

# retorna um vetor que representa a projecao vertical da imagem.
def projVertical(img):
  
    # Convert black spots to ones 
    img[img == 0]   = 1
    # Convert white spots to zeros 
    img[img == 255] = 0
  
    projecao_vertical = np.sum(img, axis = 0)
  
    return projecao_vertical

def toPNG(img, nomeImagem):
    Image.fromarray((img * 255)).save(nomeImagem + ".png")

def fromPNG(img):
    Image.fromarray((img)).save('temp.png')
    img_NC = Image.open('C:/Users/Igor/Desktop/Dropbox/6osem/pi/tp/pi-trabalho/temp.png').convert('L')
    img_NC = np.array(img_NC)
    os.remove("C:/Users/Igor/Desktop/Dropbox/6osem/pi/tp/pi-trabalho/temp.png")

    return img_NC

def inverteImagem(img):
    imagem = img.copy()    

    for idx, px in enumerate(imagem):
        if px != 0:
            imagem[idx] = -1

    for idx, px in enumerate(imagem):
        if px == 0:
            imagem[idx] = 255

    for idx, px in enumerate(imagem):
        if px == -1:
            imagem[idx] = 0

    return imagem

def main():
    
    inicializa_janela()

    """
    mndata = MNIST('samples')

    images, labels = mndata.load_training()
    # images, labels = mndata.load_testing()

    index = random.randrange(0, len(images))  # choose an index ;-)

    # print("images length = " + str(len(images)))
    # print("index = " + str(index))
    # print(mndata.display(images[index])) # printa "estilizado"

    print(images[index]) # printa em NC
    
    imagem = images[index].copy()

    for idx, px in enumerate(imagem):
        if px != 0:
            imagem[idx] = -1

    for idx, px in enumerate(imagem):
        if px == 0:
            imagem[idx] = 255

    for idx, px in enumerate(imagem):
        if px == -1:
            imagem[idx] = 0

    print()
    print(imagem)
    print()
    print()
    print("index = " + str(index))

    # converte a imagem mnist de lista p/ np.array
    imgMnist = np.array(images[6736], dtype="int")
    imgMnist = imgMnist.reshape((28,28))
    # plt.imshow(imgMnist, cmap="gray")
    # plt.show()

    imgMnist2 = np.array(imagem, dtype="int")
    imgMnist2 = imgMnist2.reshape((28,28))

    
    toPNG(imgMnist, "temp")
    toPNG(imgMnist, "exemplo-mnist")
    src = cv2.imread('C:/Users/Igor/Desktop/Dropbox/6osem/pi/tp/pi-trabalho/exemplo-mnist.png')

    # filtro gaussiano
    gaussian3 = cv2.GaussianBlur(src, (3, 3), 0) # kernel 3
    print("tipo gaussian3 = " + str(type(gaussian3)))
    print("shape gaussian3 = " + str(gaussian3.shape))

    
    
    
    # tentativa de alinhar a img: nao dá certo
    im_src = gaussian3
    pts_src = np.array([[0, 0], [0, 28], [28, 0],[28, 28]])
    im_dst = gaussian3#cv2.imread('temp.png')
    pts_dst = np.array([[14, 14], [14, 0], [0, 14], [0, 0]])
    h, status = cv2.findHomography(pts_src, pts_dst)
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    cv2.imshow("Warped Source Image", im_out)
    

    
    # filtro mediana
    mediana = cv2.medianBlur(src, 3)

    # filtro blur
    blur = cv2.blur(src, (3, 3))


    

    # printar mais de uma img na msm tela
    print("subplots")
    linhas = 5
    colunas = 2
    tela = plt.figure(figsize=(10, 5))

    tela.add_subplot(linhas, colunas, 1) # 1a imagem
    plt.imshow(imgMnist, cmap="gray")
    plt.axis("off")
    plt.title("imgMnist")

    tela.add_subplot(linhas, colunas, 2) # 2a imagem
    plt.imshow(imgMnist2, cmap="gray")
    plt.axis("off")
    plt.title("imgMnist2")

    tela.add_subplot(linhas, colunas, 3) # 3a imagem
    plt.imshow(src, cmap="gray")
    plt.axis("off")
    plt.title("imagem-png")

    #remove o "3o eixo"
    img_PNG_NC = Image.open('C:/Users/Igor/Desktop/Dropbox/6osem/pi/tp/pi-trabalho/exemplo-mnist.png').convert('L')
    # print("type of img_PNG_NC = " + str(type(img_PNG_NC)))
    img_PNG_NC = np.array(img_PNG_NC) 

    tela.add_subplot(linhas, colunas, 4) # 4a imagem
    plt.imshow(img_PNG_NC, cmap="gray")
    plt.axis("off")
    plt.title("png para NC")

    tela.add_subplot(linhas, colunas, 5) # 5a imagem
    plt.imshow(gaussian3, cmap="gray")
    plt.axis("off")
    plt.title("gaussiano kernel 3")

    img_gauss3_NC = fromPNG(gaussian3) # transforma de PNG p/ grayscale
    hist = cv2.calcHist([img_gauss3_NC], [0], None, [256], [0,256])    
    if np.argmax(hist) == 0: # se o fundo eh preto
        img_gauss3_NC = cv2.bitwise_not(img_gauss3_NC) # faz fundo branco e objeto preto
    
    limiar, imgLimiarizada = cv2.threshold(img_gauss3_NC, 0, 255, cv2.THRESH_OTSU)

    tela.add_subplot(linhas, colunas, 6) # 6a imagem
    plt.imshow(imgLimiarizada, cmap="gray")
    plt.axis("off")
    plt.title("img limiarizada - gauss kernel 3")

    tela.add_subplot(linhas, colunas, 7) # 7a imagem
    plt.imshow(mediana, cmap="gray")
    plt.axis("off")
    plt.title("mediana kernel 3")

    mediana_NC = fromPNG(mediana) # transforma de PNG p/ grayscale
    hist = cv2.calcHist([mediana_NC], [0], None, [256], [0,256])    
    if np.argmax(hist) == 0: # se o fundo eh preto
        mediana_NC = cv2.bitwise_not(mediana_NC) # faz fundo branco e objeto preto
    
    limiar, imgLimiarizadaMediana = cv2.threshold(mediana_NC, 0, 255, cv2.THRESH_OTSU)

    tela.add_subplot(linhas, colunas, 8) # 8a imagem
    plt.imshow(imgLimiarizadaMediana, cmap="gray")
    plt.axis("off")
    plt.title("img limiarizada - mediana kernel 3")

    tela.add_subplot(linhas, colunas, 9) # 9a imagem
    plt.imshow(blur, cmap="gray")
    plt.axis("off")
    plt.title("blur kernel 3")

    
    blur_NC = fromPNG(blur) # transforma de PNG p/ grayscale
    hist = cv2.calcHist([blur_NC], [0], None, [256], [0,256])    
    if np.argmax(hist) == 0: # se o fundo eh preto
        blur_NC = cv2.bitwise_not(blur_NC) # faz fundo branco e objeto preto
    
    limiar, imgLimiarizadaBlur = cv2.threshold(blur_NC, 0, 255, cv2.THRESH_OTSU)

    tela.add_subplot(linhas, colunas, 10) # 10a imagem
    plt.imshow(imgLimiarizadaBlur, cmap="gray")
    plt.axis("off")
    plt.title("img limiarizada - blur kernel 3")

    

    plt.show()
    
    print("Gaussiano:")
    hist = cv2.calcHist([img_gauss3_NC], [0], None, [256], [0,256])
    plt.plot(hist)
    plt.show()

    print("Mediana:")
    hist = cv2.calcHist([mediana_NC], [0], None, [256], [0,256])
    plt.plot(hist)
    plt.show()

    print("Blur:")
    hist = cv2.calcHist([blur_NC], [0], None, [256], [0,256])
    plt.plot(hist)
    plt.show()


    print("limiar = " + str(limiar))


    pvert = projVertical(imgLimiarizada.copy())
    phorz = projHorizontal(imgLimiarizada.copy())
    projecoes = np.concatenate((pvert, phorz))


    print(pvert)
    print(phorz)



    # seleciona imagem e a carrega com o comando "imread"
    # imgPath = fd.askopenfilename()
    # img1 = cv2.imread(imgPath) 

    # # histograma de img1
    # hist = cv2.calcHist([img1],[0],None,[256],[0,256]) 

    # # plota o histograma
    # plt.plot(hist)
    # plt.show()


    # # converte img1 em NC e guarda em img1_NC
    # img1_NC = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("img1_NC", img1_NC)

    
    # limiar, imgLimiarizada = cv2.threshold(img1_NC, 0, 255, cv2.THRESH_OTSU)

    # cv2.imshow("img1_NC", imgLimiarizada)

    # print()
    
    #printa_imagem(imgLimiarizada)
    
    #print(type(imgLimiarizada))
    """
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    janela.mainloop()

    

if __name__ == "__main__":
    main()
