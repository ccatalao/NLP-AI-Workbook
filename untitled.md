# Segmentação

Comecemos pela divisão do texto em frases e palavras. À primeira vista trata-se uma tarefa simples de automatizar. Basta para tal encontrar os elementos de separação mais comuns, como os pontos e os espaços em branco. É basicamente um processo de deteção de limites, de fronteiras. O processo é por isso frequentemente designado, em PLN, como "segmentação".  Encontramos um ponto, e teremos uma frase; encontramos um espaço, e teremos uma palavra. Duas ou três linhas de programação serão suficientes: varre-se texto, e tudo o que se encontra entre espaços é um palavra, e entre pontos é uma frase. 

  
p.p1 {margin: 0.0px 0.0px 0.0px 0.0px; font: 13.0px 'Helvetica Neue'; color: \#000000; background-color: \#ffffff}  


A segmentação de frases

  
p.p1 {margin: 0.0px 0.0px 0.0px 0.0px; font: 13.0px 'Helvetica Neue'; color: \#000000; background-color: \#ffffff}  


Num texto escrito, as frases são normalmente separadas por sinais de pontuação, como o ponto final, de interrogação ou de exclamação. As frases podem ser constituídas por uma ou por várias palavras. Mas é preciso que façam sentido. É por isso que "os pássaros cantavam na varanda" é uma frase, mas "varanda na pássaros cantavam os" não é uma frase. Como é fácil perceber, a diferença está na posição ocupada pelas palavras -- uma distinção óbvia para uma pessoa conhecedora da língua em questão, mas não para um computador. Essa informação tem de lhe ser transmitida e codificada, palavra por palavra. Essa operação é conhecida por position-of-speech tagging \(POS tagging\), isto é, anotação de posição de discurso, um assunto a que voltaremos no próximo capítulo.   

```python
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
```

