# AUXÍLIO A DEFICIENTES VISUAIS UTILIZANDO REDES NEURAIS CONVOLUCIONAIS

Este projeto propõe a extração de características utilizando redes neurais convolucionais (VGG16 + VGG19) previamente treinadas por uma base de imagem disponibilizada pelo Google (ImageNET), e a classificação utilizando o modelo de aprendizado supervisionado (SVM), a partir de um conjunto de imagens disponibilizada em tempo real por um aplicativo android que será desenvolvido a fim de capturação das imagens em tempo real.

![image](https://user-images.githubusercontent.com/89952288/205444578-8f6f77e3-dd76-470f-8ba6-1cc2b187f957.png)

O Aplicativo android possui a função de enviar imagens do caminho a ser trilhado pelo deficiente visual para a API Flask, essa API conta com duas CNNs (VGG16 + VGG19) previamente treinadas pela base de dados do Google (ImageNET), essas CNNs possuem o objetivo de extrair as caracteristicas da imagem recebida, e, com essas caracteristicas, realizar a classificação da imagem, utilizando nosso modelo de aprendizado supervisionado (SVM), também já previamente treinado. O resultado fornecido pelo SVM (0 - caminho com obstáculo | 1 - caminho livre ) será devolvido para o APP Android, que terá a responsábilidade de definir a forma de emitir essa informação para o deficiente vísual.

![image](https://user-images.githubusercontent.com/89952288/206479652-0083b75b-5974-48b5-9944-2bc1b4b14d30.png)

# Demonstração do APP:

![image](https://user-images.githubusercontent.com/89952288/206480033-77481f15-160c-4f96-acdf-ed67b941c12a.png)

# Resultados:

A partir da análise dos dados preparou-se uma matriz de confusão para demonstrar o resultado, onde os casos verdadeiros são as imagens que possuem obstáculos e falsos determinam são os de caminho livre.

![image](https://user-images.githubusercontent.com/89952288/206586160-0a499d3f-5187-4735-9c14-ce7663fbe02c.png)

Deste modo temos uma acurácia calculada de 75,3% de acertos, e uma especificidade que apura o percentual de acertos de negativos entre os casos negativos de 61,7% e uma sensibilidade que mede a taxa de acertos de casos positivos entre todas as amostras positivas de 94,9%, ou seja, o modelo proposto de acordo com os testes realizados nos demonstra uma melhor detecção em situações que a imagem capturada possui algum obstáculo. Verificando-se a eficiência que é a média entre as taxas de sensibilidade e especificidade temos uma taxa de 78,3%.

Considerando-se os métodos e abordagens utilizadas, percebeu-se que apesar da taxa de acurácia de 75,3% pode-se implementar outras abordagens de algoritmos, sensores e tecnologias para a busca de melhoria na predição desejada.

<h3>
   <br>
   Desenvolvedores: Gabirel Minguini Sanga e João Mauricio Gallego Polo
   <br>
   Orientador: Jefferson Antônio Ribeiro Passerini
</h3>
