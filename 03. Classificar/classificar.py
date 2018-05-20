
# coding: utf-8

# # LINGUAGEM NATURAL E INTELIGÊNCIA ARTIFICIAL
# 

# ## CLASSIFICAÇÃO AUTOMÁTICA DE TEXTO
'''
# #### Aplicação:
Classificar textos de acções de divulgação científica, com aplicação de aprendizagem automática supervisionada.
# Os dados utilizados neste exemplo são dois ficheiros em formato CSV:
dados_treino.csv, com textos classificados por categorias temáticas (eg., astronomia, biologia, etc)
dados_classificar.csv, com textos não classificados.
O resultado final será a criação de um ficheiro excel (dados_classificados.xls).
# #### Livrarias python
'''


import re
import pandas as pd
from collections import OrderedDict

# #### Livrarias NLTK - Natural Langage Processing Toolkit
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


# #### Livrarias SciKit Learn - Machine Learning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# #### Variáveis

corpus = []
novo_corpus = []
categorias = [ "outra","Astronomia","Biologia","Geologia","Engenharia","Patrimonio"]

TREINO = "textos_treino.csv"
CLASSIFICAR = "textos_classificar.csv"
RESULTADOS = "textos_classificados"


# #### Funções


def tokenizar(texto, stemizar=True, lingua="portuguese"):

    # Usa o SnowballStemmer do NLTK para português
    stemizador = SnowballStemmer(lingua)

    # Tokeniza por frase e por palavra
    tokens = [word.lower() for frase in nltk.sent_tokenize(texto) for word in nltk.word_tokenize(frase)]
    
    # Aplica NLTK stopwords 
    stop_words_pt = set(stopwords.words(lingua)) 
    
    # Filtra as palavras que não têm letras, e as que estão na lista de stopwords
    tokens_filtrados = []
    for token in tokens:
        if re.search("[a-zA-Z]", token):
            if token not in stopwords.words(lingua):
                if token not in stop_words_pt:
                    tokens_filtrados.append(token)
                if stemizar:
                    stems = [stemizador.stem(token) for token in tokens_filtrados]
    return stems



# Importando os dados de treino
dados_treino = pd.read_csv(TREINO, delimiter = "\t", quoting = 3)


# #### Pré-processamento dos textos de treino
print("\nA treinar com textos classificados ...\n")
for i in range(len(dados_treino)):    
    termos = tokenizar(dados_treino["texto"][i], stemizar=True, lingua="portuguese")    
    # refaz cada linha com os tokens processados
    termos = ' '.join(termos)
    corpus.append(termos)


# #### Calcular a matriz TF-IDF

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

# #### Importar novos textos para classificação
dados_classificar = pd.read_csv(CLASSIFICAR, delimiter = "\t", quoting = 3)

# #### Pré-processamento dos novos textos
print("A classificar novos textos ...\n")
for i in range(len(dados_classificar)):    
    termos = tokenizar(dados_classificar["texto"][i], stemizar=True, lingua="portuguese")    
    # refaz cada linha com os termos processados
    termos = ' '.join(termos)
    novo_corpus.append(termos)


# #### Calcular a matiz TF-IDF dos novos textos
X_novo = cv.transform(novo_corpus)
X_novo_tfidf = tfidf_transformer.transform(X_novo)


# #### Previsão de classificações e respectivas probabilidades
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



# #### Guardar os novos textos, já classificados

# Criar o dataframe com os textos classificados
df = pd.DataFrame(nova_classificacao)
# Manter a ordem das colunas no dataframe 
df = df[[ k for k in nova_classificacao[0].keys()]] 

# Salvar num ficheiro CSV
df.to_csv(RESULTADOS + ".csv", index=False, sep='\t')

# Salvar num ficheiro excel
df.to_excel(RESULTADOS + ".xlsx", index=False)

print("Resultados gravados em excel e csv.\n")


