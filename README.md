# AUXÍLIO A DEFICIENTES VISUAIS UTILIZANDO REDES NEURAIS CONVOLUCIONAIS

Este projeto propõe a extração de características utilizando redes neurais convolucionais (VGG16 + VGG19) previamente treinadas por uma base de imagem disponibilizada pelo Google (ImageNET), e a classificação utilizando o modelo de aprendizado supervisionado (SVM), a partir de um conjunto de imagens disponibilizada em tempo real por um aplicativo android que será desenvolvido a fim de capturação das imagens em tempo real.

![image](https://user-images.githubusercontent.com/89952288/205444578-8f6f77e3-dd76-470f-8ba6-1cc2b187f957.png)

O Aplicativo android possui a função de enviar imagens do caminho a ser trilhado pelo deficiente visual para a API Flask, essa API conta com duas CNNs (VGG16 + VGG19) previamente treinadas pela base de dados do Google (ImageNET), essas CNNs possuem o objetivo de extrair as caracteristicas da imagem recebida, e, com essas caracteristicas, realizar a classificação da imagem, utilizando nosso modelo de aprendizado supervisionado (SVM), também já previamente treinado. O resultado fornecido pelo SVM (0 - caminho com obstáculo | 1 - caminho livre ) será devolvido para o APP Android, que terá a responsábilidade de definir a forma de emitir essa informação para o deficiente vísual.

![image](https://user-images.githubusercontent.com/89952288/205410525-66777348-a071-4858-9e85-be4062c3e7fc.png)

# Demonstração do APP:

<h3>Caminho limpo:</h3>

![image](https://user-images.githubusercontent.com/89952288/205452056-bfa88d0e-a690-439b-99be-64312b5eb2d3.png)

<br><h3>Caminho com obstáculo:</h3>

![image](https://user-images.githubusercontent.com/89952288/205452015-aa1069c5-1b8d-4655-933e-037cba7186b1.png)

<br>
Desenvolvedores: Gabirel Minguini Sanga e João Mauricio Gallego Polo
<br>
Orientador: Jefferson Antônio Ribeiro Passerini
