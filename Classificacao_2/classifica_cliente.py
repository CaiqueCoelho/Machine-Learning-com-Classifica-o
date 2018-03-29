#from dados import carregar_alura
import pandas as pd

#contador inteligente do python
from collections import Counter


def fit_and_predict(modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes, nome_do_modelo):

	#treina o nosso modelo baseado nos nossos dados
	modelo.fit(treino_dados, treino_marcacoes)

	#tenta prever o resultado com os testes
	resultado = modelo.predict(teste_dados)

	#diferencas = resultado - teste_marcacoes para 0 ou 1 no resultado
	acertos = (resultado == teste_marcacoes)

	#acertos = [d for d in diferencas if d == True] #para 0 ou 1 no resultado
	total_de_acertos = sum(acertos) #len(acertos) para 0 ou 1 no resultado
	total_de_elementos = len(teste_dados)
	taxa_de_acerto = 100.0 * total_de_acertos/total_de_elementos

	print("Taxa de acerto do algoritmo " +nome_do_modelo + " foi de: " + str(taxa_de_acerto) + "% " + "de " + str(total_de_elementos) + " elementos")

	return taxa_de_acerto

#data_frame, pandas devolve
df = pd.read_csv('cliente.csv')

X_df = df[['recencia', 'frequencia', 'semanas_de_inscricao']]
Y_df = df['situacao']

# Transforma as variaveis categoricas em variaveis binarias
Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

# Transforma de data_frames para arrays
X = Xdummies_df.values
Y = Ydummies_df.values

#Treino do algorimo
porcentagem_de_treino = 0.8
porcentagem_de_teste = 0.1

#treino 0:799
tamanho_de_treino = porcentagem_de_treino * len(Y)
treino_dados = X[:int(tamanho_de_treino)]
treino_marcacoes = Y[:int(tamanho_de_treino)]

#teste 800:899
tamanho_de_teste = len(Y) * porcentagem_de_teste
fim_de_teste = tamanho_de_treino + tamanho_de_teste
teste_dados = X[int(tamanho_de_treino):int(fim_de_teste)]
teste_marcacoes = Y[int(tamanho_de_treino):int(fim_de_teste)]

#validacao 900:999
validacao_dados = X[int(fim_de_teste):]
validacao_marcacoes = Y[int(fim_de_teste):]

resultados = {}

#Um versos o resto/todo (One versus Rest/All)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
#0 => 0 e 1 => 1,2, LinearSVC vai falar que eh do tipo 0 ou do resto (38%, resto 62%)
#0 => 0,2 e 1 => 1 LinearSVC vai falar que eh do tipo 1 ou do resto (44%, resto 56%)
#0 => 0,1 e 2 => 2, LinearSVC vai falar que eh do tipo 0 ou do resto (20%, restp80%)
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0))
resultadoOneVsRest = fit_and_predict(modeloOneVsRest, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes, "One Vs Rest Classifier")
resultados[resultadoOneVsRest] = modeloOneVsRest

#OneVsOne
from sklearn.multiclass import OneVsOneClassifier
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0))
resultadoOneVsOne = fit_and_predict(modeloOneVsOne, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes, "One Vs One")
resultados[resultadoOneVsOne] = modeloOneVsOne

#importa o algoritmo de classidicacao MultinomialNB
from sklearn.naive_bayes import MultinomialNB
#atribui o algoritmo MultinomialNB a variavel chamada modelo
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict(modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes, "MultinomialNB")
resultados[resultadoMultinomial] = modeloMultinomial

#Algotimo que encontra a melhor possibilidade de combinacoes das caracteristicas dos dados de treino
#AdaBoost foi igual ao MultinomialNB para alura2.csv (82%), mas foi melhor para alura.csv (85% contra 82%)
from sklearn.ensemble import AdaBoostClassifier
#atribui o algoritmo AdaBoostClassifier a variavel chamada modelo
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict(modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes, "AdaBoostClassifier")
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

print resultados
maximo = max(resultados)

vencedor = resultados[maximo]
print vencedor

resultado = vencedor.predict(validacao_dados)

#diferencas = resultado - teste_marcacoes para 0 ou 1 no resultado
acertos = (resultado == validacao_marcacoes)

#acertos = [d for d in diferencas if d == True] #para 0 ou 1 no resultado
total_de_acertos = sum(acertos) #len(acertos) para 0 ou 1 no resultado
total_de_elementos = len(validacao_marcacoes)
taxa_de_acerto = 100.0 * total_de_acertos/total_de_elementos

print("Taxa de acerto do algoritmo melhor no mundo real" + " foi de: " + str(taxa_de_acerto) + "% " + "de " + str(total_de_elementos) + " elementos")
