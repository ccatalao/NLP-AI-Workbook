#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Programa de classificação automática de texto.
# @author: ccatalao
# Carlos Catalão Alves 

# Criado em 13 Maio 2018
# Última actualização: 2018-05-20 18:36 (Dom, 20 Maio 2018)
 


# #### Aplicação:

# Programa que classifica automaticamente acções de divulgação científica, com aplicação de aprendizagem automática supervisionada.  
#   
#   

# Os textos são lidos em dois ficheiros em formato csv, delimitados por tabs:  
# 
# 1. treinar.csv (com textos classificados, para treino do modelo)   
# 2. classificar.csv (com os novos textos para classificação automática) 
# 
# O resultado final é a criação de dois ficheiros com os novos textos já classificados,   
# com indicação da respectiva probabilidade:
# 
# 1. classificados.csv (ficheiro csv, delimitado com tabs)  
# 2. classifcados.xlsx (ficheiro excel)
# 
# 


import re
import pandas as pd
from collections import OrderedDict
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# Variáveis
corpus = []
novo_corpus = []
categorias = [ "outra","Astronomia","Biologia","Geologia","Engenharia","Patrimonio"]
TREINO = "treinar.csv"
CLASSIFICAR = "classificar.csv"
RESULTADOS = "classificados"


# Funções

def atomizar(texto, radical=True, lingua="portuguese"):

    # usa o SnowballStemmer do NLTK para português para obter o radical
    radicalizador = SnowballStemmer(lingua)

    # atomiza por frase e palavra
    atomos = [word.lower() for frase in nltk.sent_tokenize(texto) for word in nltk.word_tokenize(frase)]
    
    # Aplica NLTK stopwords 
    stop_words_pt = set(stopwords.words(lingua)) 
    
    # Filtra as palavras que não têm letras, e as que estão na lista de stopwords
    atomos_filtrados = []
    for atomo in atomos:
        if re.search("[a-zA-Z]", atomo):
            if atomo not in stopwords.words(lingua):
                if atomo not in stop_words_pt:
                    atomos_filtrados.append(atomo)
                if radical:
                    radicais = [radicalizador.stem(atomo) for atomo in atomos_filtrados]
    return radicais


# Ler os dados de treino
dados_treino = pd.read_csv(TREINO, delimiter = "\t", quoting = 3)

# Pré-processamento dos textos de treino
print("\nA treinar com textos classificados ...\n")
for i in range(len(dados_treino)):    
    atomos = atomizar(dados_treino["texto"][i], radical=True, lingua="portuguese")    
    # refaz cada linha com os tokens processados
    atomos = ' '.join(atomos)
    corpus.append(atomos)

# Calcular a matriz TF-IDF
# Vectorizar só as 100 palavras mais frequentes
cv = CountVectorizer(max_features = 100)

# Obter a variável independente X (ie., os textos já classificados)
X = cv.fit_transform(corpus).toarray()
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)

# Obter a variável independente y (a classificação dos textos)
# ":" todas as linhas, e "2" a coluna com a classific
y = dados_treino.iloc[:, 2].values 

# Aplicar o modelo Multinominal Naive Bayes
clf = MultinomialNB().fit(X_tfidf, y)

# Ler novos textos para classificação
dados_classificar = pd.read_csv(CLASSIFICAR, delimiter = "\t", quoting = 3)

# Pré-processamento dos novos textos

print("A classificar novos textos ...\n")
for i in range(len(dados_classificar)):    
    atomos = atomizar(dados_classificar["texto"][i], radical=True, lingua="portuguese")    
    # refaz cada linha com os termos processados
    atomos = ' '.join(atomos)
    novo_corpus.append(atomos)

# Calcular a matiz TF-IDF dos novos textos
X_novo = cv.transform(novo_corpus)
X_novo_tfidf = tfidf_transformer.transform(X_novo)

# Previsão de classificações e respectivas probabilidades

previsoes = clf.predict(X_novo_tfidf)
probabilidades = clf.predict_proba(X_novo_tfidf)

nova_classificacao = []
for i in range(len(dados_classificar)): 
    p = previsoes[i]
    #pb = p - 1 # Obter a coluna com a probabilidade válida

    line = {
            "titulo":dados_classificar["titulo"][i],
            "texto":dados_classificar["texto"][i],
            "previsao":p,
            "categoria":categorias[previsoes[i]],
            #p - 1 para a coluna com a probabilidade válida
            "probabilidade":(probabilidades[i][p-1]) * 100,             
            }
    OrderedDict(line) # para preservar a ordem das colunas
    nova_classificacao.append(line)

# Criar o dataframe com os textos classificados
df = pd.DataFrame(nova_classificacao)
# Manter a ordem das colunas no dataframe 
df = df[[ k for k in nova_classificacao[0].keys()]] 

# Salvar num ficheiro CSV
df.to_csv(RESULTADOS + ".csv", index=False, sep='\t')

# Salvar num ficheiro excel
df.to_excel(RESULTADOS + ".xlsx", index=False)

print("Resultados gravados em excel e csv.\n")

