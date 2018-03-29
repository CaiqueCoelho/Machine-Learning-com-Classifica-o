#!-*- coding: utf8 -*-

import pandas as pd
from collections import Counter
import numpy as np
from sklearn.cross_validation import cross_val_score

texto1 = "Se eu comprar cinco anos antecipados, eu ganho algum desconto?"
texto2 = "O exercício 15 do curso de Java 1 está com a resposta errada. Pode conferir por favor?"
texto3 = "Existe algum curso para cuidar do marketing da minha empresa"

def vetorizar_texto(texto, tradutor):
	vetor = [0] * len(tradutor)

	for palavra in texto:
		if palavra in tradutor:
			posicao = tradutor[palavra]
			vetor[posicao] += 1

	return vetor

classificacoes = pd.read_csv("emails.csv")
textosPuros = classificacoes['email']
#coloco todo mundo em minusculo para nao diferenciar, maiusculo e divido palavra por palavra 
textosQuebrados = textosPuros.str.lower().str.split(' ')

#set() para trasnformar a variavel dicionario em um conjunto, ou seja, um "array" que não contem elementos iguais
dicionario = set()
for lista in textosQuebrados:
	dicionario.update(lista)

totalDePalavras = len(dicionario)
#zip combina em tuplas o que eu quero do lado esquerdo e do lado direito
tuplas = zip(dicionario, xrange(totalDePalavras))

tradutor = {palavra:indice for palavra, indice in tuplas}

#print totalDePalavras

print vetorizar_texto(textosQuebrados[0], tradutor)
vetoresDeTexto = [vetorizar_texto(texto, tradutor) for texto in textosQuebrados]

marcacoes = classificacoes['classificacao']

X = np.array(vetoresDeTexto)
Y = np.array(marcacoes.tolist())

porcentagem_de_treino = 0.8

tamanho_do_treino = porcentagem_de_treino * len(Y)
tamanho_de_validacao = len(Y) - tamanho_do_treino

treino_dados = X[0:int(tamanho_do_treino)]
treino_marcacoes = Y[0:int(tamanho_do_treino)]

validacao_dados = X[int(tamanho_do_treino):]
validacao_marcacoes = Y[int(tamanho_do_treino):]

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
	k = 10
	scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv = k)
	taxa_de_acerto = np.mean(scores)
	
	msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
	print msg
	return taxa_de_acerto

resultados = {}

#Um versos o resto/todo (One versus Rest/All)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
#0 => 0 e 1 => 1,2, LinearSVC vai falar que eh do tipo 0 ou do resto (38%, resto 62%)
#0 => 0,2 e 1 => 1 LinearSVC vai falar que eh do tipo 1 ou do resto (44%, resto 56%)
#0 => 0,1 e 2 => 2, LinearSVC vai falar que eh do tipo 0 ou do resto (20%, restp80%)
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0))
resultadoOneVsRest = fit_and_predict("One Vs Rest Classifier", modeloOneVsRest, treino_dados, treino_marcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest

from sklearn.multiclass import OneVsOneClassifier
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0))
resultadoOneVsOne = fit_and_predict("One Vs One", modeloOneVsOne, treino_dados, treino_marcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne

#importa o algoritmo de classidicacao MultinomialNB
from sklearn.naive_bayes import MultinomialNB
#atribui o algoritmo MultinomialNB a variavel chamada modelo
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial

#Algotimo que encontra a melhor possibilidade de combinacoes das caracteristicas dos dados de treino
#AdaBoost foi igual ao MultinomialNB para alura2.csv (82%), mas foi melhor para alura.csv (85% contra 82%)
from sklearn.ensemble import AdaBoostClassifier
#atribui o algoritmo AdaBoostClassifier a variavel chamada modelo
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

#A eficacia do algoritmo que chuta tudo 0 ou 1 ou um unico valor
#usando array
#acerto_de_um = len(Y[Y=='sim']) #ou sum(Y)
#acerto_de_zero = len(Y[Y=='nao']) #ou len(Y) - sum(Y)
#usando lista 
acerto_base = max(Counter(validacao_marcacoes).itervalues()) #Devolve a quantidade do maior elemento
acerto_de_um = list(Y).count('sim')
acerto_de_zero = list(Y).count('nao')
#taxa_de_acerto_base = 100.0 * max(acerto_de_um, acerto_de_zero) / len(Y)
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base nos dados de validacao: %f" %taxa_de_acerto_base)

#print resultados
maximo = max(resultados)

vencedor = resultados[maximo]
print vencedor

vencedor.fit(treino_dados, treino_marcacoes)
resultado = vencedor.predict(validacao_dados)

#diferencas = resultado - teste_marcacoes para 0 ou 1 no resultado
acertos = (resultado == validacao_marcacoes)

#acertos = [d for d in diferencas if d == True] #para 0 ou 1 no resultado
total_de_acertos = sum(acertos) #len(acertos) para 0 ou 1 no resultado
total_de_elementos = len(validacao_marcacoes)
taxa_de_acerto = 100.0 * total_de_acertos/total_de_elementos

print("Taxa de acerto do algoritmo melhor no mundo real" + " foi de: " + str(taxa_de_acerto) + "% " + "de " + str(total_de_elementos) + " elementos")
