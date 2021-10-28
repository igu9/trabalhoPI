# Trabalho Prático - Parte 1
# Disciplina: Processamento de Imagens
# Professor: Alexei Manso Correa Machado
# Grupo:
# Igor Marques Reis
# Lucas Spartacus Vieira Carvalho
# Rafael Mourão Cerqueira Figueiredo

import pt2 # parte 2 do trabalho (classificar os digitos)
import os, math, time, random, cv2 #opencv
import numpy as np
import tkinter as tk
from matplotlib import pyplot as plt
from tkinter import filedialog as fd
# from tkinter import messagebox
from PIL import ImageTk, Image  # pip install Pillow
from mnist import MNIST

# variáveis globais
imgPath = None
img = None
imagem_limiarizada = None

janela = tk.Tk()
lblImg = tk.Label(image = "")
largura = 800
altura = 600

projecoes_treino_mnist = []
projecoes_teste_mnist = []
projecoes_digitos = []

# Metodo que permite exibir a imagem na interface grafica
def toPNG(img, nomeImagem):
    cv2.imwrite(nomeImagem, img)

# Redimensiona imagem para que ela caiba na interface grafica
def redimensiona_imagem(imagem, width=504, height=504):
    
    if imagem.width > width:
        if imagem.height > height:
            imagem = imagem.resize((width, height))
        else: imagem = imagem.resize((width, imagem.height))
    else:
        if imagem.height > height:
            imagem = imagem.resize((imagem.width, height))

    return imagem

# "Plota" a imagem na interface grafica
def plota_img(imagem_npArray):
    toPNG(imagem_npArray, "tmp.png")

    imagem_tk = Image.open(os.getcwd()+"\\tmp.png")

    if imagem_tk.width > 504 or imagem_tk.height > 504:
        imagem_tk = redimensiona_imagem(imagem_tk)
    imagem_tk = ImageTk.PhotoImage(imagem_tk)
    
    
    global lblImg
    lblImg.config(image = "")
    lblImg = tk.Label(image = imagem_tk)
    lblImg.image = imagem_tk
    lblImg.place(x = 15, y = 50)

# Utiliza Otsu para limiarizar a imagem
def tira_limiar(inv):
    global imagem_limiarizada
    img_limiar = cv2.imread(imgPath)
    img_limiar_NC = cv2.cvtColor(img_limiar, cv2.COLOR_BGR2GRAY)

    # filtro gaussiano
    img_filtroGauss = cv2.GaussianBlur(img_limiar_NC.copy(), (5, 5), 0) # kernel 5

    if inv: limiar, im = cv2.threshold(img_filtroGauss.copy(), 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    else: limiar, im = cv2.threshold(img_filtroGauss.copy(), 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    imagem_limiarizada = im.copy()
    plota_img(im)

# Recorta cada digito da imagem, os exibe e tira suas projecoes
def acha_contorno():
    global img, imagem_limiarizada
    
    contornos, _ = cv2.findContours(imagem_limiarizada.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img2 = imagem_limiarizada.copy()
    digitos = []
    for c in contornos:
        x,y,w,h = cv2.boundingRect(c)
        
        # Faz retangulo delimitando cada digito
        cv2.rectangle(img2, (x,y), (x+w, y+h), color=(255, 255, 255), thickness=2)
        
        # Guarda posicao do digito na imagem
        digito = imagem_limiarizada[y:y+h, x:x+w]
        
        # Redimensiona o digito p/ (18, 18)
        digito_redimensionado = cv2.resize(digito, (18,18))
        
        # Adiciona 5px de margem para adequar o digito ao padrao mnist
        digito_com_margem = np.pad(digito_redimensionado, ((5,5),(5,5)), "constant", constant_values=0)
        
        digitos.append(digito_com_margem)
    
    # converte para array numpy
    digitos_np = np.array(digitos)
    num_digitos = len(digitos_np)
    
    # mostra os digitos
    numLinhas = num_digitos/2
    numColunas = num_digitos/2

    plt.rcParams["figure.figsize"] = [5.0, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rc('axes.spines',top=False,bottom=False,left=False,right=False)
    plt.rc('axes',facecolor=(1,1,1,0),edgecolor=(1,1,1,0))
    plt.rc(('xtick','ytick'),color=(1,1,1,0))

    for idx, digito in enumerate(digitos_np, 1):
        plt.subplot(numLinhas, numColunas, idx)
        plt.imshow(digito, cmap="gray")
    plt.show()

    # tira projecoes dos digitos
    global projecoes_digitos
    projecoes_digitos = []
    for i in range(num_digitos):
        projecoes_digitos.append(np.concatenate((projHorizontal(digitos_np[i].copy()), projVertical(digitos_np[i].copy()))))

    print(digitos_np[5])
    print(projecoes_digitos[5])

# Obtem path da imagem e a exibe na interface
def carregar_imagem():
    global imgPath
    imgPath = fd.askopenfilename()
    print(imgPath)

    global img
    img = cv2.imread(imgPath)

    plota_img(img)

# Exclui o arquivo temp ao fechar o programa
def on_closing():
    if os.path.isfile("tmp.png"):
        os.remove("tmp.png")
    janela.destroy()

# Seta parametros da interface grafica
def inicializa_janela():

    # parametros da interface grafica
    janela.geometry(str(largura) + "x" + str(altura))
    janela.columnconfigure(0, weight=1)
    janela.rowconfigure(0, weight=1)
    janela.title("OCR")
    
    janela.protocol("WM_DELETE_WINDOW", on_closing) # confirmacao ao fechar a janela

    # botoes e rotulos - declaracao
    btnCarregarImg = tk.Button(janela, text = "Carregar imagem", bg = "#7E7E7E", fg = "#ffffff", font = ("Calibri", 10), command = lambda : carregar_imagem())
    btnLimiar = tk.Button(janela, text = "Limiariza imagem", bg = "#7E7E7E", fg = "#ffffff", font = ("Calibri", 10), command = lambda : tira_limiar(False))
    btnLimiarInvertido = tk.Button(janela, text = "Limiariza imagem - invertido ", bg = "#7E7E7E", fg = "#ffffff", font = ("Calibri", 10), command = lambda : tira_limiar(True))
    btnRecortaDigitos = tk.Button(janela, text = "Recortar dígitos", bg = "#7E7E7E", fg = "#ffffff", font = ("Calibri", 10), command = lambda : acha_contorno())

    # botoes e rotulos - posicao
    btnCarregarImg.place(x = 15, y = 20)
    btnLimiar.place(x = 135, y = 20)
    btnLimiarInvertido.place(x = 260, y = 20)
    btnRecortaDigitos.place(x = 450, y = 20)

# retorna um vetor que representa a projecao horizontal da imagem
def projHorizontal(img):
  
    return np.sum(img, axis = 1) 

# retorna um vetor que representa a projecao vertical da imagem.
def projVertical(img):
  
    return np.sum(img, axis = 0)


def main():
    global projecoes_treino_mnist, projecoes_teste_mnist

    inicializa_janela()

    
    mndata = MNIST('samples')

    mnist_treino, lbl_mnist_treino = mndata.load_training()
    mnist_teste, lbl_mnist_teste = mndata.load_testing()

    projecoes_treino_mnist = []
    projecoes_teste_mnist = []

    print()
    
    # converte imagens para numpy.array
    # aplica filtro e tira suas projecoes
    for i in range(len(mnist_treino)):
        mnist_treino[i] = np.array(mnist_treino[i], dtype="uint8").reshape((28,28))
        # mnist_treino[i] = 255 - mnist_treino[i]
        mnist_treino[i] = cv2.GaussianBlur(mnist_treino[i], (3,3), 0) # filtro gaussiano, kernel 3
        projecoes_treino_mnist.append(np.concatenate((projHorizontal(mnist_treino[i].copy()), projVertical(mnist_treino[i].copy()))))


    for i in range(len(mnist_teste)):
        mnist_teste[i] = np.array(mnist_teste[i], dtype="uint8").reshape((28,28))
        # mnist_teste[i] = 255 - mnist_teste[i]
        mnist_teste[i] = cv2.GaussianBlur(mnist_teste[i], (3,3), 0) # filtro gaussiano, kernel 3
        projecoes_teste_mnist.append(np.concatenate((projHorizontal(mnist_teste[i].copy()), projVertical(mnist_teste[i].copy()))))
    

    pt2.svm(np.array(mnist_teste), np.array(projecoes_treino_mnist), np.array(lbl_mnist_treino), 
            np.array(mnist_teste), np.array(projecoes_teste_mnist), np.array(lbl_mnist_teste))




    # idxtreino = random.randrange(0, len(mnist_treino))
    # idxteste = random.randrange(0, len(mnist_teste))
    # plt.imshow(mnist_treino[idxtreino], cmap="gray")
    # plt.show()
    # plt.imshow(mnist_treino[idxteste], cmap="gray")
    # plt.show()



    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    janela.mainloop()

    

if __name__ == "__main__":
    main()