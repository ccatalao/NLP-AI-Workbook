# Segmentação

Comecemos pela divisão do texto em frases e palavras. À primeira vista trata-se uma tarefa simples de automatizar. Basta para tal encontrar os elementos de separação mais comuns, como os pontos e os espaços em branco. É basicamente um processo de deteção de limites, de fronteiras. O processo é por isso frequentemente designado, em PLN, como "segmentação".  Encontramos um ponto, e teremos uma frase; encontramos um espaço, e teremos uma palavra. Duas ou três linhas de programação serão suficientes: varre-se texto, e tudo o que se encontra entre espaços é um palavra, e entre pontos é uma frase. 

