#x dados que eu tenho e y lado que queremos prever
import csv

def carregar_acessos():

	X = []
	Y = []

	arquivo = open('acesso_pagina.csv', 'rb')
	leitor = csv.reader(arquivo)

	leitor.next()

	for home, como_funciona, contato, comprou in leitor:

		dado = [int(home), int(como_funciona), int(contato)]

		X.append(dado)
		Y.append(int(comprou))

	return X, Y

def carregar_alura():

	X = []
	Y = []

	arquivo = open('alura.csv', 'rb')
	leitor = csv.reader(arquivo)

	leitor.next()

	for home, busca, logado, comprou in leitor:

		dado =  [int(home), busca, int(logado)]

		X.append(dado)
		Y.append(int(comprou))

	return X, Y