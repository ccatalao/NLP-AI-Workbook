#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Programa que cria um corpus não anotado a partir de um ficheiro csv delimitado por tabs
#
# Criado em Maio 12 16:10:08 2018
#
# @author: ccatalao


import pandas as pd
import nltk


FICHEIRO = "imprensa.csv"
CORPO = "corpo.txt"
GENERO = "noticia"
TEXTO_INICIO = "<texto id="
TEXTO_FIM = "\n</texto>\n"
TITULO_INICIO = "<t> "
TITULO_FIM = " </t>"
PARAGRAFO_INICIO = "\n<p>"
PARAGRAFO_FIM = "\n\n</p>"
FRASE_INICIO = "\n<s> "
FRASE_FIM = " </s>"


# recebe os dados do ficheiro fonte
dados = pd.read_csv(FICHEIRO, delimiter = "\t", quoting = 3)

# abre um ficheiro de texto para registar o corpus
corpus = open(CORPO, "w")  

# usa o separador frásico para português do NLTK
separador_frasico = nltk.data.load("tokenizers/punkt/portuguese.pickle")

# para cada linha do ficheiro fonte
for indice in range(len(dados)):

    tx = "" # texto
    p = [] # lista de parágrafos
    pos_ficheiro = 2 # coluna do ficheiro fonte com a data 
    
    # regista cabeçalho do texto com metadados
    data = str(dados.iloc[indice, pos_ficheiro])
    data = data[6:8]+"-"+data[4:6]+"-" + data[:4]
    corpus.write(TEXTO_INICIO + str(indice) + " " + "gen=" + GENERO + " " 
                 + "data=" + data + " " + "origem=Ciência Viva na Imprensa Regional" + ">")
    
    # regista o título
    pos_ficheiro = 3 # coluna do ficheiro fonte com a data
    corpus.write(PARAGRAFO_INICIO)
    corpus.write(FRASE_INICIO + TITULO_INICIO + dados.iloc[indice, pos_ficheiro] 
                 + TITULO_FIM + FRASE_FIM)
    corpus.write(PARAGRAFO_FIM)     
           
    # recebe texto do ficheiro fonte
    pos_ficheiro = 4 # coluna do ficheiro fonte com o texto 
    tx = dados.iloc[indice, 4]  
    
    # separa os parágrafos
    p = tx.split('\n\n') 
        
    # processa cada parágrafo
    for x in range (0, len(p)):
        
        if p[x] != " ": # se o parágrafo não é uma linha em branco
                        
            corpus.write(PARAGRAFO_INICIO)            
            
            frases = separador_frasico.tokenize(p[x])
            
            for frase in frases:
                corpus.write(FRASE_INICIO + frase + FRASE_FIM)
            
            corpus.write(PARAGRAFO_FIM)

    corpus.write(TEXTO_FIM)

corpus.close() # fecha o ficheiro com o corpus

