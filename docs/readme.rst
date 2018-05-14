
# Capítulo 2
##O pré-processamento  
  

  
####Introdução  


Quando processamos um texto escrito em linguagem natural fazemo-lo para revelar padrões capazes de permitir aos computadores aprender a realizar de forma eficaz funções com impacto no dia-a-dia, como tradução automática, reconhecimento de voz, ou outras. Contudo, como veremos neste capítulo, a capacidade dos computadores para aprender com dados linguísticos é extremamente limitada. Para que estes aprendam a partir de um texto, é imprescindível transformá-lo em material digerível por motores matemáticos que só aceitam números como combustível. O texto, no seu estado primário (o chamado raw texto), embora eloquente para um humano, é de pouca utilidade para uma máquina.   

Para tal o texto tem de ser previamente processado. Na maior parte dos casos será necessário dividi-lo em frases, repartir as frases em palavras, anotar cada uma delas, testar combinações de palavras, remover as pouco significativas e, finalmente, reduzir cada uma das restantes a matrizes de dígitos, zero ou um. 

Dependendo do uso que pretendemos dar à aprendizagem da máquina, será preciso executar, no todo ou em parte, cada uma destas operações - algumas simples, outras não tanto. Nos primórdios do Processamento de Linguagem Natural (PNL) estas operações eram executadas com corpus linguísticos convencionais, relativamente pequenos, corretamente escritos, num muito reduzido número de línguas. Hoje, com a internet e a consequente extensão incomensurável de dados textuais, produzidos a cada minuto por milhões de utilizadores em todo o mundo, torna-se indispensável o uso de técnicas automáticas que são dificilmente aplicáveis da mesma forma no imenso espectro de línguas faladas e escritas no planeta.   

A complexidade é, portanto, a regra. A aparente facilidade dos exemplos que se seguem não deve por isso, ofuscar a verdadeira dimensão dos desafios enfrentados pelo processamento de linguagem natural. Desafios que, embora fora do âmbito deste livro, não podem deixar de ser considerados e remetidos para outras obras de referência nestas matérias.  

A decomposição de um texto nos seus elementos significativos mais elementares, a que nos habituámos desde o bancos da escola,  é agora indispensável para o processamento automático de linguagem. O conjunto destas operações e das respetivas técnicas constituí o que se designa por pré-processamento. É o que vamos aprender a fazer ao longo deste capítulo, especialmente aplicado a uma única língua: o português.  

###1. Segmentação  

Comecemos pela divisão do texto em frases e palavras. À primeira vista trata-se uma tarefa simples de automatizar. Basta para tal encontrar os elementos de separação mais comuns, como os pontos e os espaços em branco. É basicamente um processo de deteção de limites, de fronteiras. O processo é por isso frequentemente designado, em PLN, como "segmentação".  Encontramos um ponto, e teremos uma frase; encontramos um espaço, e teremos uma palavra. Duas ou três linhas de programação serão suficientes: varre-se texto, e tudo o que se encontra entre espaços é um palavra, e entre pontos é uma frase. 
 
####1.2  A segmentação de frases  

Num texto escrito, as frases são normalmente separadas por sinais de pontuação, como o ponto final, de interrogação ou de exclamação. As frases podem ser constituídas por uma ou por várias palavras. Mas é preciso que façam sentido. É por isso que "os pássaros cantavam na varanda" é uma frase, mas "varanda na pássaros cantavam os" não é uma frase. Como é fácil perceber, a diferença está na posição ocupada pelas palavras -- uma distinção óbvia para uma pessoa conhecedora da língua em questão, mas não para um computador. Essa informação tem de lhe ser transmitida e codificada, palavra por palavra. Essa operação é conhecida por position-of-speech tagging (POS tagging), isto é, anotação de posição de discurso, um assunto a que voltaremos no próximo capítulo.   

		>> import nltk
		>> sent_tokenizer=nltk.data.load('tokenizers/punkt/  
			portuguese.pickle')
		>> texto = "Esta é a primeira frase, a frase 1.   
			Depois temos a segunda frase, a frase 2. 
			E agora a terceira."
		>> frases = sent_tokenizer.tokenize(texto)
		>> for frase in frases:
           print(frase)

	Esta é a primeira frase, a frase 1. 
	Depois temos a segunda frase, a frase 2. 
	E agora a terceira.

Por enquanto, esqueçamos que uma frase tem de fazer sentido. Fiquemos só pelos elementos delimitadores da frase. Dizíamos atrás que um ponto servia como separador entre frases. Mas só se for um ponto final: o ponto em "Sr." não vale como separador, é simplesmente uma forma de sinalizar abreviaturas, tão comuns na escrita. Mais uma vez, o computador não sabe como distinguir um ponto final de um ponto de abreviatura; ou sabe? Talvez se lho dissermos. Recorremos para tal a uma lista de abreviaturas, tão completa quanto possível. E para complicar ainda mais as coisas, não chega uma lista, temos de ter várias, uma para cada língua ("Sr." não é o mesmo que "Mr."). E o computador terá de ser devidamente programado para verificar essas listas, abreviatura por abreviatura, sempre que encontrar um ponto. A tal simples linha de programação de que falávamos há pouco já não parece tão simples.  

Mas não desesperemos, há soluções. Podemos sempre ensinar a máquina a identificar a língua em que o texto está escrito, e também a distinguir o ponto final. E é aqui que começamos a perceber, na prática, a utilidade dos corpora, da anotação morfológica ou da aprendizagem das máquinas. A ligação entre processamento de linguagem natural e inteligência artificial começa a tomar forma.  Uma das técnicas usadas para ensinar o computador será, neste caso, treiná-lo com um corpus previamente anotado linguisticamente, testá-lo em novas palavras, e finalmente gravar a informação resultante, incluindo o modelo estatístico usado na aprendizagem. Com esta informação, bastará de facto uma simples linha de programação. A caixa "Experimentar" mostra como podemos utilizar as ferramentas do NLTK para separar as frases de um texto escrito em português.

####1.3  A segmentação de palavras  

**A tokenização**  

A decomposição do texto em palavras - tokenização ou atomização - é, porventura, a mais corrente das operações em tratamento automático de texto. E é assim porque os tokens daí resultantes são a matéria-prima de todas as fases subsequentes do processamento de linguagem natural, da anotação linguística à extração de termos e interpretação da informação. 
Uma das razões para dividir o texto em palavras (tokens) é também o fato da grande maioria das técnicas de aprendizagem de máquina (machine learning) partirem da análise estatística da frequência com que as palavras surgem nos textos. Se tomarmos como exemplo a classificação automática de documentos, como no caso dos artigos de divulgação científica, é muito provável que os que se debruçam sobre astronomia tenham o maior número de ocorrências de palavras como telescópio ou de combinações de palavras, como observação astronómica.
A divisão de uma frase em palavras não é tão trivial como à primeira vista possa parecer, dependendo do que se entende por palavras, ou da forma como estas são segmentadas. Esse entendimento não é necessariamente consensual, e pode mudar consoante a língua em questão, já que nem em todas as línguas as palavras são segmentadas por espaços. Mas mesmo naquelas em que o são, como é o caso do português, a tokenização encerra uma série de desafios. Vejamos alguns deles.

	>> import nltk
	>> frases = " Esta é a primeira frase, a frase 1. 
		Depois temos a segunda frase, a frase 2. E agora a terceira."
	>> palavras = nltk.word_tokenize(frases)
	>> print(palavras)

 	['Esta', 'é', 'a', 'primeira', 'frase', ',', 'a', 'frase', '1',  
 	 '.', 'Depois', 'temos', 'a', 'segunda', 'frase', ',', 'a',  
 	  'frase', '2', '.', 'E', 'agora', 'a', 'terceira', '.']

Se o príncipio básico da tokenização é o de considerar como um token qualquer caracter ou sequência de caracteres que se encontre entre dois espaços, o que fazer então com os sinais de pontuação, como o ponto, a vírgula ou o ponto e virgula? Para muitas operações de processamento automático de linguagem natural é indispensável considerar estes sinais como tokens em si mesmos, que devem ser isolados e contabilizados como tal.  

A hifenização é outra fonte de problemas, sobretudo em línguas como a portuguesa, onde cumpre funções distintas. Surge muito frequentemente nas formas reflexivas (pertence-me), na formação de tempos verbais  (tornar-se-á), mas também na quebra das palavras no final da linha quando estas são demasiado compridas para aí caberem por inteiro - e, neste último caso, é ainda mais complicado porque o hífen pode ocupar posições diferentes numa mesma palavra consoante o espaço disponível na linha.  
 
Um outro desafio é o uso do apóstrofe, raro no português, mas abundante no francês e no inglês. Um programa de tokenização de textos em inglês deve, por exemplo, expandir I'm em dois tokens distintos (I e am), ou o we'll em we e will. O mesmo acontece na língua francesa com c'est, l'age, j'ai, e em tantas outras contrações que requerem desdobramento em vários tokens.   

Há ainda o caso das palavras compostas, que constituem grande parte do vocabulário da língua alemã, como Krankenhaus (hospital, casa dos doentes) ou Schwarzwald (Floresta Negra). Estruturas semelhantes de aglutinação surgem também no português, sendo a composição marcada pelo hífen, como em circum-navegação ou super-homem. Aqui, por exemplo, o tokenizador tem de tomar decisões sobre considerar uma ou duas palavras.  
 
Temos ainda o caso das chamadas EM (Expressões Multipalavra), muito frequentes na língua portuguesa e que constituem, por isso, um dos mais sérios obstáculos ao rigor do processamento de linguagem natural na nossa língua. Não se trata de expandir um token em dois, mas de decidir como contrair num único token um grupo composto por várias palavras. Acresce ainda que estamos perante expressões que estão em processo de criação constante, a maior parte das vezes como resultado da apropriação social da terminologia científica, como em aquecimento global ou inteligência artificial.   
Os métodos computacionais utilizados para lidar com estas expressões requerem muitas vezes a aplicação de algoritmos de aprendizagem de máquina, havendo já investigação recente sobre esta matéria (ref?). Uma técnica de uso mais comum é a utilização de n-grams, que veremos em detalhe mais à frente.  

Os sistemas de processamento de linguagem mais primitivos, nos primeiros tempos da PLN, não davam grande importância a estas questões, consideradas então como pouco relevantes. A segmentação de textos em fases e palavras não parecia levantar grandes problemas. Os textos a processar eram pequenos, e as suas aplicações finais também. À tradução automática, por exemplo, bastava-lhe produzir frases curtas e pouco rigorosas - a tolerância era grande, o que se poderia esperar de uma máquina? Por  outro lado, os corpus linguísticos a processar eram compostos por textos jurídicos, literários ou científicos, escritos e corrigidos com o maior rigor e consistência. A maior parte das vezes já tinham sido anotados manualmente por linguistas profissionais e técnicos competentes. A ambiguidade, tão característica da linguagem, estava  já esclarecida por anotações sintáticas, morfológicas e lexicais.  
 
Hoje a situação é bem diferente. O tempo é de Big Data, os corpos medem-se aos milhões de palavras e alimentam-se na internet, nos chats, nos comentários de utilizadores da imprensa on-line, nos newsgroups, nas redes sociais, onde a escrita é errática, a pontuação inconsistente, as palavras reduzidas a abreviaturas, e as frases polvilhadas de códigos HTML, endereços de email, urls incompreensíveis, smiles e hastags; e onde a tolerância dos consumidores para os erros das aplicações das tecnologias da linguagem é, como no caso dos corretores ortográficos, praticamente zero.   

Quando se trata de tokenização, o tratamento computacional de textos é uma sucessão constante de tomadas de decisão. Incluir ou não incluir pontuação, isolar ou não partes de uma palavra, partir ou não uma palavra em diferentes tokens, combinar ou não várias palavras num único token, expandir ou não as palavras contraídas. Estas são apenas algumas das questões mais frequentes. E as respostas nunca são universais. O que está bem para o inglês pode não estar para o francês. O que serve para uma aplicação, como a tradução automática, pode não servir para a aprendizagem de máquina. O que funciona num corpus escrito com todas as regras da ortografia e da gramática pode não funcionar num corpus de posts e tweets, onde essas regras nem sempre se aplicam.   

Mas a capacidade de processamento dos computadores também aumentou exponencialmente. O que então exigia processamento paralelo em linhas de computadores é hoje possível num portátil de última geração. As possibilidades do processamento de linguagem natural estão agora em aberto com os avanços na capacidade de aprendizagem das máquinas e na investigação em sistemas inteligentes.  

**Normalização**  

A normalização implica um conjunto de operações que podem ser executadas no todo ou em parte, consoante o propósito do processamento. Quando o objectivo não é centrado no processamento linguístico, com anotação morfo-sintáctica, podemos recorrer a formas de normalização e limpeza que passam pela eliminação de números, sinais de pontuação e passagem a minúsculas. Estas formas de normalização são sobretudo utilizadas para efeitos de contabilização de ocorrências de palavras, cuja estatística é fundamental para funções de aprendizagem automática, já que não faz muito sentido está a contabilizar de forma diferente uma mesma palavra quando esta surge com letra minúscula e maiúscula. Como podemos ver a seguir, números e pontos finais foram eliminados na tokenização, e todas as palavras começam com letra minúscula.


	>> import nltk
	>> frases = " Esta é a primeira frase, a frase 1. Depois temos a  
	 segunda frase, a frase 2. E agora a terceira."
	>> palavras = nltk.word_tokenize(frases)
	>> palavras = [palavras.lower() for palavras in palavras   
		if  palavras.isalpha()]
	>> print(palavras)

	['esta', 'é', 'a', 'primeira', 'frase', 'a', 'frase', 'depois',   
	'temos', 'a', 'segunda', 'frase', 'a', 'frase', 'e', 'agora',   
	'a', 'terceira']


####1.4  A aplicação de filtros  

**Usando stop words em português**  

A expressão stop words tornou-se comum no mundo dos motores de pesquisa, e refere-se habitualmente às palavras de uso comum numa dada língua. Todos temos já a experiência da inutilidade de fazer uma pesquisa usando termos como "de" ou "com". Isto porque a maior parte dos motores de pesquisa pura e simplesmente remove estas palavras dos seus índices, não só porque são inúteis no processo de pesquisa mas também porque ocupam espaço de memória e tempo de processamento nos índices utilizados para acelerar as pesquisas.
No ponto de vista do processamento de linguagem natural colocam-se as mesmas desvantagens de tempo, espaço e esforço de programação. A não ser que haja uma razão específica - normalmente de natureza linguística - para manter estas palavras comuns, é preferível eliminá-las durante o processamento. Como o podemos fazer?  

 A forma mais simples é utilizar uma lista de stop words. É claro que é impossível determinar com absoluta exactidão quais são, de facto, as palavras mais usadas numa dada língua. Existem, por isso, muitas listas para cada língua. Mas a verdade é que cada um de nós pode criar a sua própria lista, ou alterar as listas existentes, juntando ou removendo palavras dessas listas.  
 
Existem, naturalmente, métodos mais sofisticados e elegantes, automáticos e dinâmicos (isto é, que assinalam e gerem as stop words à medida que as vão detectando por métodos estatísticos. Um método que se tornou muito popular (também a partir dos motores de pesquisa) é a criação de índices baseados no valor de TF-IDF, isto porque os termos de busca mais úteis são os que são mais frequentes (valor elevado de TF, term frequency), mas só num número reduzido de documentos (valor elevado de IDF, inverse document frequency).   
Voltaremos a esta técnica mais à frente, a propósito de aplicações de inteligência artificial na aprendizagem automática.
Para já, voltemos ao método mais simples, o das listas fixas de stop words.   

O NLTK tem uma dessas listas, que vamos usar aqui no nosso exemplo (notar que o artigo "a" é removido após a aplicação da lista stopwords.words('portuguese'):  

	>> import nltk
	>> from nltk.corpus import stopwords
	>> stop_words = set(stopwords.words('portuguese'))
	>> frases = "Esta é a primeira frase, a frase 1."
	>> palavras = nltk.word_tokenize(frases)
	>> palavras = [palavras.lower() for palavras in palavras   
		if palavras not in stop_words]
	>> print(palavras)

	['esta', 'é', 'primeira', 'frase', ',', 'frase', '1', '.']



####1.5  Extracção de estemas - a stemização  

A stemização (do inglês, stemming) é uma das operações de preprocessamento de texto mais eficazes para recuperação de informação, classificação ou tradução automática. É, contudo, um procedimento que, além de complexo, dá lugar a alguns mal-entendidos quanto à sua definição, alcance e aplicação.   

O primeiro mal-entendido a evitar é o de pensar a stemização como a extracção da raiz morfológica de uma palavra. Na verdade, trata-se simplesmente de uma técnica de redução de palavras ao menor número de caracteres, aplicando algoritmos de  extracção de um tronco (stem) comum que permita agrupar palavras relacionadas. Para, quando estamos perante variações morfológicas de uma palavra que comportam um conteúdo semântico semelhante (e.g., conhecer, conhecimento, conhecido), o algoritmo de stemização liga todas estas palavras ao mesmo termo (e.g., conhec) . Deste modo, assumindo que as palavras com o mesmo tronco partilham a mesma ideia ou conceito, podemos agrupá-las sob o mesmo estema ou termo de indexação, aumento deste modo a eficiência dos processos de pesquisa e de recuperação de informação.  

Um segundo mal-entendido é pensar que a stemização tem em conta o significado das palavras. Na verdade, trata-se de um processo de aplicação de regras muito básicas de construção de palavras, e varia substancialmente com a língua, pelo que está sujeito a erros (ver a este propósito Hanbook of Natural Language Processing, 2010: 468). Esta é uma das razões porque se utiliza, em alternativa, o processo de lematização - uma técnica de determinação da forma canónica da palavra como lema num dicionário ou lista de vocabulário (ver caixa stemização vs lematização).

Caixa stemização vs lematizaçãoLematização está intimamente ligada à stematização. A diferença 	é que esta última opera numa palavra sem conhecimento do 	contexto, e por isso não consegue distinguir palavras que têm 	diferentes significados dependendo do contexto e posição (pos-tagging). Contudo, stemizadores são mais rápidos, mais fáceis de implementar, e a falta de precisão pode não contar para algumas aplicações).  

Por exemplo:
A palavra "foi" tem "ser" como lema. Esta ligação é perdida com a stemização, porque requer a verificação num dicionário.
A palavra "passear" é a base da palavra "passearam", e por isso é detectada tanto na stemização como na lematização.

Existe, assim, um leque variado de ferramentas de stemização, com graus de precisão diferentes consoante os métodos utilizados e as línguas a que se aplicam.   

Para efeitos de simplificação vamos usar aqui o stemizador do NLTK para a língua portuguesa.

	>> import nltk
	>> from nltk.stem.snowball import SnowballStemmer
	>> stemmer = SnowballStemmer('portuguese')
	>> palavras = ['conhecer', 'conhecimento', 'conhecido', 'amor',  
	 'amoroso', 'amores']
	>> estemas = [stemmer.stem(palavras) for palavras in palavras]
	>> print(estemas)

	['conhec', 'conhec', 'conhec', 'amor', 'amor', 'amor']

####2. Listas e frequência de palavras

Um dos propósitos do pré-processamento é a criação de listas de palavras a partir de um texto, com indicação da respectiva frequência. Esta secção será dedicada a este objectivo, aproveitando para tal para criar uma livraria de funções que permitam aplicar as etapas que estudámos até aqui.

Para percebermos melhor como funciona o calculo de frequência de palavras vamos utilizar as funções que o NLTK coloca ao nosso dispor para esse fim. 

	>> import nltk
	>> frases = "Esta é a primeira frase, a frase 1.   
		Depois temos a segunda frase, a frase 2. E agora a terceira."
	>> palavras = nltk.word_tokenize(frases)
	>> palavras = [palavras.lower() for palavras in palavras   
	if palavras.isalpha()]
	>> palavras_frequentes = nltk.FreqDist(palavras)
	>> for key in palavras_frequentes:
            print(key + ' - ' + str(palavras_frequentes[key]))

	esta - 1
	é - 1
	a - 5
	primeira - 1
	frase - 4
	depois - 1
	temos - 1
	segunda - 1
	e - 1
	agora - 1
	terceira - 1


