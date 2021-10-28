# Trabalho Prático - Parte 2
# Disciplina: Processamento de Imagens
# Professor: Alexei Manso Correa Machado
# Grupo:
# Igor Marques Reis
# Lucas Spartacus Vieira Carvalho
# Rafael Mourão Cerqueira Figueiredo

import sys, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from datetime import datetime
from tensorflow import keras

def formata_tempo(tempo_execucao):
    tempo_formatado = ""
    aux = ["h", "min"]
    cont = 0
    for i, c in enumerate(tempo_execucao):
        if c == ':':
            tempo_formatado += aux[cont]
            cont += 1
        else : tempo_formatado += tempo_execucao[i]
    tempo_formatado += 's'

    return tempo_formatado

# proj_digitos -> lista das projecoes (horiz. e vert.) dos digitos
def svm(digitos_teste, proj_digitos_teste, rotulos_teste, digitos_treino, proj_digitos_treino, rotulos_treino):
    t_inicio = datetime.now()

    x_teste, y_teste = keras.utils.normalize(proj_digitos_teste[0:30000]), rotulos_teste[0:30000]

    funcTransformacao = "poly"
    svm = SVC(gamma = 0.00001, kernel = funcTransformacao, C = 100)



    svm.fit(x_teste, y_teste) # treina a svm
    #49742
    #10663

    # idx = random.randrange(0, len(digitos_teste)) # seleciona um indice aleatorio
    # print("indice = ", idx)
    

    # digito_predito = svm.predict(np.reshape(proj_digitos_teste[11080], (1,-1))) # testar apenas 1 digito
    digitos_preditos = svm.predict( keras.utils.normalize(proj_digitos_treino) )
    print("digitos_preditos len = ", len(digitos_preditos))
    

    acuracia = metrics.accuracy_score(rotulos_treino, digitos_preditos)
    matriz_confusao = metrics.confusion_matrix(rotulos_treino, digitos_preditos)
    print("acc ({}) = {}".format(funcTransformacao, acuracia))

    t_fim = datetime.now()

    print()
    tempo_formatado = formata_tempo(str(t_fim-t_inicio))
    print("datetime - tempo de execução = ", tempo_formatado)



def main():
    # svm()
    print("main")
    # digitos, rotulos_teste = sys.argv[1], sys.argv[2]
    


if __name__ == '__main__':
    main()