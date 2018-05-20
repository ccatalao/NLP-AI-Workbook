#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Programa de classificação automática de texto.
# @author: ccatalao
# Carlos Catalão Alves 

# Criado em 13 Maio 2018
# Última actualização: 2018-05-20 18:36 (Dom, 20 Maio 2018)
 

# #### Aplicação:

# Obter e comparar os resultados de diferentes modelos de classificação.  
#   
# Os modelos comparados são:
#
#   k-Nearest Neighbor (kNN)
#   SGDClassifier
#   Perceptron
#   Naive Bayes (modelos Multinominal e Bernoulli)

# Os dados são lidos num ficheiro csv, delimitado por tabs (textos_treino.csv), 
# e são divididos em séries para treino e para teste. Os resultados são obtidos 
# a partir de uma comparação entre as classificações previstas pelos modelos 
# treinados e as classificações reais à partida.
# 


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk

from time import time
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import PlaintextCorpusReader, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics 


# #### Funções

# Relatório de classificação
def reporta(clf, modelo):
    
    print("\nModelo: %s" % modelo)
    print('_' * 80)

    clf.fit(X_treino, y_treino)
    previsao = clf.predict(X_teste)
    score = metrics.accuracy_score(y_teste, previsao)
    print("\nacurancy: %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("\nTermos mais relevantes em cada categoria:\n")
        for i, label in enumerate(categorias):
            top10 = np.argsort(clf.coef_[i])[-10:]
            print(trim("%s: \t %s" % (label, "  ".join(termos[top10]))))
        print()


    print("\nRelatório:")
    print(metrics.classification_report(y_teste, previsao, target_names=categorias))
    print("\nconfusion matrix:")
    print(metrics.confusion_matrix(y_teste, previsao))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score

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

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

# #############################################################################

report = []
categorias = [ "Astronomia","Biologia","Geologia","Engenharia","Patrimonio"]

dados = pd.read_csv('treinar.csv', delimiter = '\t', quoting = 3)

X = dados["texto"]
y = dados["target"]
# Dividir documentos em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

print("%d documentos de treino" % (len(X_treino)))
print("%d documentos de teste" % (len(X_teste)))
print("%d categorias" % len(categorias))
print()


print("A extrair termos de %d documentos de treino ..."  % (len(X_treino)))
t0 = time()
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', tokenizer=atomizar)
X_treino = vectorizer.fit_transform(X_treino)
duracao = time() - t0
print("Termos extraídos em: {0:.1f}s".format(duracao))
print("documentos: %d, termos: %d" % X_treino.shape)
print()

print("A extrair termos de %d documentos de teste ..."  % (len(X_teste)))
t0 = time()
X_teste = vectorizer.transform(X_teste)
duracao = time() - t0
print("Termos extraídos em: {0:.1f}s".format(duracao))
print("documentos: %d, termos: %d" % X_teste.shape)
print()

# Ligar cada vector ao termo original, para imprimir os termos mais relevantes)
termos = vectorizer.get_feature_names()
if termos:
    termos = np.asarray(termos)



# Perceptron
print('=' * 80)
from sklearn.linear_model import Perceptron
report.append(reporta(Perceptron(max_iter=50), "Perceptron"))

# KNN
print('=' * 80)
from sklearn.neighbors import KNeighborsClassifier
report.append(reporta(KNeighborsClassifier(n_neighbors=10), "KNeighbors Classifier"))


# SGD 
print('=' * 80)
from sklearn.linear_model import SGDClassifier
report.append(reporta(SGDClassifier(alpha=.0001, max_iter=50,
                                       penalty="elasticnet"), "SGD Classifier"))


# Naive Bayes
print('=' * 80)
print("Naive Bayes")
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
report.append(reporta(MultinomialNB(alpha=.01), "MultinomialNB"))
report.append(reporta(BernoulliNB(alpha=.01), "BernoulliNB"))



# Gráfico comparativo de resultado
indices = np.arange(len(report))
report = [[x[i] for x in report] for i in range(2)]
clf_names, score = report
plt.figure(figsize=(8, 4))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)
plt.show()
