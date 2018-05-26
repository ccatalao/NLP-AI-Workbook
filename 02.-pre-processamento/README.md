# 02. Pre-processamento

Quando processamos um texto escrito em linguagem natural fazemo-lo para revelar padrões capazes de permitir aos computadores aprender a realizar de forma eficaz funções com impacto no dia-a-dia, como tradução automática, reconhecimento de voz, ou outras. Contudo, como veremos neste capítulo, a capacidade dos computadores para aprender com dados linguísticos é extremamente limitada. Para que estes aprendam a partir de um texto, é imprescindível transformá-lo em material digerível por motores matemáticos que só aceitam números como combustível. O texto, no seu estado primário \(o chamado raw texto\), embora eloquente para um humano, é de pouca utilidade para uma máquina.   

Para tal o texto tem de ser previamente processado. Na maior parte dos casos será necessário dividi-lo em frases, repartir as frases em palavras, anotar cada uma delas, testar combinações de palavras, remover as pouco significativas e, finalmente, reduzir cada uma das restantes a matrizes de dígitos, zero ou um. 

Dependendo do uso que pretendemos dar à aprendizagem da máquina, será preciso executar, no todo ou em parte, cada uma destas operações - algumas simples, outras não tanto. Nos primórdios do Processamento de Linguagem Natural \(PNL\) estas operações eram executadas com corpus linguísticos convencionais, relativamente pequenos, corretamente escritos, num muito reduzido número de línguas. Hoje, com a internet e a consequente extensão incomensurável de dados textuais, produzidos a cada minuto por milhões de utilizadores em todo o mundo, torna-se indispensável o uso de técnicas automáticas que são dificilmente aplicáveis da mesma forma no imenso espectro de línguas faladas e escritas no planeta.   

A complexidade é, portanto, a regra. A aparente facilidade dos exemplos que se seguem não deve por isso, ofuscar a verdadeira dimensão dos desafios enfrentados pelo processamento de linguagem natural. Desafios que, embora fora do âmbito deste livro, não podem deixar de ser considerados e remetidos para outras obras de referência nestas matérias.  

A decomposição de um texto nos seus elementos significativos mais elementares, a que nos habituámos desde o bancos da escola,  é agora indispensável para o processamento automático de linguagem. O conjunto destas operações e das respetivas técnicas constituí o que se designa por pré-processamento. É o que vamos aprender a fazer ao longo deste capítulo, especialmente aplicado a uma única língua: o português.  



