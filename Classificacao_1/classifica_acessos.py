from dados import carregar_acessos
from sklearn.naive_bayes import MultinomialNB

X, Y = carregar_acessos()

#Minha abordagem foi
#1. Separar 90% para treino e 10% para teste: 88.89%

treino_dados = X[:90]
treino_marcacoes = Y[:90]

teste_dados = X[90:] #ou X[-9:] ultimas nove linhas
teste_marcacoes = Y[90:] #ou Y[-9:] ultimas nove linhas

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo. predict(teste_dados)
diferencas = resultado - teste_marcacoes

acertos = [d for d in diferencas if d == 0]

total_acertos = len(acertos)
total_elementos = len(teste_dados)

taxa_acerto = 100.0 * total_acertos/total_elementos

print(str(taxa_acerto) + '%')
print("de")
print(total_elementos)
