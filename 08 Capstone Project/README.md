# Projeto final de curso

Programa que detecta uma mão e conta o número de dedos levantados.

Faremos um polígono ao redor da mão do usuário como mostra a foto
![hand](hand_convex.png)

Mediremos os pontos dentro desse polígono a partir do "centro" da mão
* Adotaremos o "centro" como a distância média entre os ponto mais distantes na horizontal e na vertical

Não contaremos mudanças nos pontos mais baixos (pulsos) nem em locais distantes do polígono (objs fora da mão)
