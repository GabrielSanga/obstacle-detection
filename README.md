# AUXÍLIO A DEFICIENTES VISUAIS UTILIZANDO REDES NEURAIS CONVOLUCIONAIS

Este projeto propõe a extração de características utilizando redes neurais convolucionais (VGG16 + VGG19) previamente treinadas por uma base de imagem disponibilizada pelo Google (ImageNET), e a classificação utilizando o modelo de aprendizado supervisionado (SVM), a partir de um conjunto de imagens disponibilizada em tempo real por um aplicativo android que será desenvolvido a fim de capturação das imagens em tempo real.

![image](https://user-images.githubusercontent.com/89952288/205444578-8f6f77e3-dd76-470f-8ba6-1cc2b187f957.png)

O Aplicativo android possui a função de enviar imagens do caminho a ser trilhado pelo deficiente visual para a API Flask, essa API conta com duas CNNs (VGG16 + VGG19) previamente treinadas pela base de dados do Google (ImageNET), essas CNNs possuem o objetivo de extrair as caracteristicas da imagem recebida, e, com essas caracteristicas, realizar a classificação da imagem, utilizando nosso modelo de aprendizado supervisionado (SVM), também já previamente treinado. O resultado fornecido pelo SVM (0 - caminho com obstáculo | 1 - caminho livre ) será devolvido para o APP Android, que terá a responsábilidade de definir a forma de emitir essa informação para o deficiente vísual.

![image](https://user-images.githubusercontent.com/89952288/205410525-66777348-a071-4858-9e85-be4062c3e7fc.png)

Demonstração do APP:

<br>
Demonstração de uso real:
Caminho limpo:
![WhatsApp Image 2022-12-03 at 13 32 30](https://user-images.githubusercontent.com/89952288/205451509-cb888eac-84ee-4a10-8c63-98993e3cc16e.jpeg)

<br>
Caminho com obstáculo:
![WhatsApp Image 2022-12-03 at 13 32 12](https://user-images.githubusercontent.com/89952288/205451517-704776f6-2df2-4919-beec-82fc0b4fba1a.jpeg)

<br>
Desenvolvedores: Gabirel Minguini Sanga e João Mauricio Gallego Polo
<br>
Orientador: Jefferson Antônio Ribeiro Passerini
