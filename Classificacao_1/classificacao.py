#import numpy as np
#import matplotlib.pyplot as plt

#Algoritmo de classificacao mais simples
from sklearn.naive_bayes import MultinomialNB

#gordinho, perninha curta, faz auau
porco1 = 	[1, 1, 0]
porco2 = 	[1, 1, 0]
porco3 = 	[1, 1, 0]
cachorro1 = [1, 1, 1]
cachorro2 = [0, 1, 1]
cachorro3 = [0, 1, 1]

dados =[porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

#1: cachorro, -1: cachorro
marcacoes = [1, 1, 1, -1, -1, -1]

modelo = MultinomialNB()
#Funcao fit faz modelo se adequar as informacoes(dados e marcacoes)
modelo.fit(dados, marcacoes)

misterioso1 = [1, 1, 1]
misterioso2 = [1, 0, 0]
misterioso3 = [0, 0, 1]
misterioso4 = [1, 0, 1]
testes = [misterioso1, misterioso2, misterioso3, misterioso4]

marcacoes_teste = [-1, 1, -1, 1]

resultado = modelo.predict(testes)
print(resultado)

diferencas = resultado - marcacoes_teste

acertos =  [diferenca for diferenca in diferencas if diferenca == 0]

total_de_acertos = len(acertos)
total_de_elementos = len(testes)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
print(str(taxa_de_acerto) + "%")

#Preve se misterioso eh cachorro ou porquinho
#print(modelo.predict(teste))