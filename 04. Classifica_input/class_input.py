#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Programa de classificação automática de texto.
# @author: ccatalao
# Carlos Catalão Alves 

# Criado em 28 Outubro 2018
# Última actualização: 2018-10-28 18:36 (Dom, 28 Outubro 2018)
 


# #### Aplicação:

# Programa que classifica automaticamente uma frase introduzida pelo utilizador,
#   

# O modelo é treinado com um ficheiro .csv (treinar.csv), a partir de uma tabela 
# criada para o efeito com dados da Ciência Viva no Verão

# 
# O resultado final é a indicação da categoria/área de actividade, com a respectiva probabilidade:
# 
# 


import re
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# Variáveis
corpus = []
novo_corpus = []
input_corpus = []
categorias = [ "outra","Astronomia","Biologia","Geologia","Engenharia","Patrimonio"]
TREINO = "treinar.csv"



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
# ":" todas as linhas, e "2" a coluna com a classificação
y = dados_treino.iloc[:, 2].values 

# Aplicar o modelo Multinominal Naive Bayes
clf = MultinomialNB().fit(X_tfidf, y)

         
while True:

    try: 
        # Calcular a matiz TF-IDF do novo texto
        my_input = input("\nEscreva uma frase para classificar -> ")
        input_corpus = []
        novo_corpus = []
        
        input_corpus.append(my_input)
        
        print("\nA processar a frase ...\n")
        atomos = atomizar(input_corpus[0], radical=True, lingua="portuguese")    
        # refaz linha com os termos processados
        atomos = ' '.join(atomos)
        novo_corpus.append(atomos)
        
        
        X_novo = cv.transform(novo_corpus)
        X_novo_tfidf = tfidf_transformer.transform(X_novo)
        
        # Previsão de classificações e respectivas probabilidades
        
        previsoes = clf.predict(X_novo_tfidf)
        probabilidades = clf.predict_proba(X_novo_tfidf)
        
        previsao = previsoes[0]
        categoria = categorias[previsao]
        probabilidade = (probabilidades[0][previsao-1]) * 100
        
        if probabilidade > 50: 
        
            print(novo_corpus[0] + '\n')
            print("\nCategoria: %s. Probabilidade = %f" % (categoria, probabilidade))
        else:
            print("\nNão consigo classificar essa frase com um grau de confiança superior a 50%. Tente de novo ...\n")
            
    except(KeyboardInterrupt, EOFError, SystemExit):
        break
