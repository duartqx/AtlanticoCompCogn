# Projeto 2: Bootcamp Atlântico Academy - Computação Cognitiva - Squad III

Relatório sobre o segundo projeto para o bootcamp do Instituto Atlântico trilha Computação Cognitiva. Utilizando ferramentas como opencv-python ou scikit-image, bastante conhecidas em trabalhos de processsamento digital de imagens, este projeto tinha como objetivo segmentar um dataset de cem folhas de plantas que o grupo deveria coletar e fotografar seguindo um padrão definido pelo próprio grupo.

O trabalho foi dividido em três etapas, sendo elas:
1. Definir um padrão ouro com vinte das cem folhas, suas segmentações verdadeiras e métricas de maior/menor comprimento e suas áreas.
2. Segmentar o padrão ouro utilizando três métodos diferentes de utilizando scikit-image e calcular qual desses métodos apresentavam o melhor resultado IOU (Intersection over Union) comparado as segmentações verdadeiras (Pintadas manualmente com o Krita, um conhecido programa de produção de imagens open-source disponível gratuitamente para os principais sistemas operacionais).
3. Com a melhor segmentação, baseada no resultado do IOU e testes, segmentar todas as outras imagens, calcular suas métricas e exportá-las em um csv, além de salvar a imagem original lado a lado com sua segmentação.

## Gold Standard

### Padrão definido pelo Squad 

- Fotografia de folhas de plantas variadas em cima de uma folha de papel tamanho A4 na posição retrato. 
- Fotografar com o flash ligado para diminuir as sombras
- Enquadramento mostrando o A4 por completo e possivelmente com poucas bordas além dele
- Se possível, autocrop com a camera do célular para as bordas do A4

A primeira etapa de pré-processamento utilizada nas fotografias foi usar o programa de linha de comando `convert`, pertencente à suite de aplicações ImageMagick, para redimensionar todas as imagens de gold standard para 900px de altura para diminuir o tempo de processamento entre cada uma durante as etapas de segmentação automática e análise de métricas.

Inicialmente imaginamos que era obrigatório que as imagens do padrão outro tivessem suas segmentações verdadeiras feitas exclusivamente com algum método do scikit-image ou opencv, e para isso foi escolhido o método limiar por isodata somado a `binary_dilation` e `binary_fill_holes`, que trazia bons resultados em grande quantidade de folhas. Um grande problema que encontramos com este método foi por ele não realizar segmentações perfeitas, mesmo após muitos testes e alterações, já que em algumas fotografias com muito brilho a resultava em segmentações com muitos buracos que muitas vezes não podiam ser fechados usando funções prontas como `binary_dilation`+`scipy.ndimage.binary_fill_holes`. Um tutorial sobre como encontrar a melhor segmentação na documentação do scikit-image trazia a informação que para encontrar a melhor era necessário comparar com a ouro/segmentação verdadeira. Surgiu aí a dúvida sobre se o nosso método "isodata" serviria como ouro, sabendo que algumas segmentações não estavam perfeitas, e que ela na verdade deveria ser comparada com outra segmentação superior. No mesmo tutorial sobre encontrar a melhor segmentação incluía a informação que é comum o padrão ouro ser feito manualmente, isso respondeu a principal dúvida sobre o padrão ouro e foi o que produzimos em seguida. Usando o software krita segmentamos manualmente com ajuda de uma tablet wacom todos os padrões ouro verdadeiros.

Com o padrão ouro em mãos e a segmentação por isodata já feita, segmentamos as folhas do padrão ouro mais algumas vezes com outros métodos, e os finais escolhidos foram os métodos que chamamos de canny, flood_fill e o já mencionado isodata. Apesar desses três métodos terem nomes conhecidos, eles na verdade são a combinação de ao menos três em conjunto, como várias passagens por `binary_dilation`, `binary_fill_holes` e também `remove_small_objects` para limpar as segmentações de qualquer pontos indesejáveis.

## Segmentação dos Padrões Ouro

Antes da proposta do projeto já tinhamos iniciado os trabalhos em uma classe com métodos de segmetação utilizando scikit-image. Continuamos trabalhando nesta codebase com algumas refatoraçôes e acrescentando novos métodos e melhor organização no módulo segmentation.py. Essa classe, chamada Segment, recebe obrigatoriamente uma string que deve ser o caminho para uma imagem, ela automaticamente carrega essa imagem utilizando a função `skimage.io.imread`, uma versão preto e branco da imagem original e uma versão com gaussian blur aplicado na versão preto e branca, caso seja necessária em atributos da classe. A classe continua com a definição de vários métodos de segmentação, como os já citados isodata, canny, flood_fill, mas também vários outros tipos de limiar, ict, chanvese, sauvola, entre outras.

### Métodos de segmentação

1. **Isodata**

Utilizando `skimage.filters.thresholding.try_all_threshold` que testa vários tipos de limiares para análise subjetiva de qual é o melhor, o limiar que apresentava melhor resultado nas imagens utilizadas como padrão ouro o era o limiar por isodata. Isodata resultou em segmentações geralmente bem limpas, mas imperfeitas, enquanto muitos dos outros tipos de limiar faziam sombras nas fotografias ficarem completamente pretas, e o método limiar local em especial que nas nossas fotografias resultava em imagens totalmente ou quase totalmente pretas.

2. **Canny**

Canny é um método de detecção de bordas que trás resultados geralmente muito bons e detalhados, mas nas nossas fotos era muito comum essas bordas não fecharem completamente, por isso aplicamos algumas vezes a função `skimage.morphology.binary_dilation` e após isso foi aplicado a função `scipy.ndimage.binary_fill_holes` que preenche totalmente formas fechadas, resultando em uma segmentação com alto nível de precisão na maioria das imagens. O grande problema desse método foi com folhas muito claras, com partes brancas ou folhas grande demais, que passavam um pouco do tamanho do papel A4, o que resultava nas bordas detectadas nesses casos não serem fechadas e `binary_fill_holes` não poder preencher elas.

3. **Flood fill**

Sem dúvidas este foi um dos métodos mais divertidos de escrever. A função `flood_fill` do `skimage` precisa de uma seed, uma tuple de dois int que representa o pixel no `pd.ndarray` (row, column), que é a imagem, que `flood_fill` vai começar a atuar. Como cada imagem tinha uma folha de espécie diferente, tamanhos diferentes, e nenhuma perfeitamente centralizada na folha de papel A4, por era impossível escolher um ponto fixo para ser a seed. Para resolver este problema, após buscas, escrevemos um pequeno staticmethod que recebe a imagem como argumento, um pd.ndarray, e a transforma em um array de uma única dimensão com o método flatten do próprio array e com a função min() o menor valor nesse array, em seguida utilizando np.where procuramos o index desse menor valor no ndarray original da imagem e retornamos uma tuple de dois int que é o index (row, column) do pixel mais escuro na imagem. Com essa seed em mãos aplicamos flood_fill com tolenrância padrão de 120, o que se provou nas 20 imagens ouro suficiente para pintar toda a folha sem tocar na cor do fundo. Primeiro usamos 255 como valor para completamente pintar as folhas com flood_fill, mas isto acabou não funcionando bem com algumas das fotografias não ouro que tinham o o fundo quase totalmente brancos, em contraste com a outra metade das fotografias que o fundo não eram tão perfeitos. Para resolver este problema, invertemos a cor do flood_fill de 255 para 0. Para extrair somente uma imagem binária desse flood_fill usamos uma expressão booleana de igualdade para o cor aplicada. E `scipy.ndimage.binary_fill_holes` para tentar evitar buracos facilmente removíveis nas folhas.

4. **Outros testes**

Também realizamos testes com os métodos de segmentação Iterative Cluster Threshold, Felzenszwalb, Chanvese e outros. Visualmente todos esses apresentavam resultados muito interessantes. Ict em especial recebeu notas altíssimas com sua precisão avaliada, O seu grande problema é que ele se mostrou um pouco arisco e com frequentes falhas em segmentar bem todas as folhas, algumas muito pequenas acabavam desaparecendo em vez de serem segmentadas completamente ou maiores que se unia ao fundo branco em pontas aleatórias, distorcendo a forma verdadeira da folha. Muito tempo foi utilizado tentando encontrar um balanceamento nos dois principais argumentos da função usada para gerar ict, mas ainda não encontramos os valores ideais. Chanvese chamou atenção com seus resultados esteticamente interessantes, mas ele se mostrou um algoritmo muito demorado e com resultados as vezes completamente inúteis para as fotografias que tinhamos em mãos para serem o padrão ouro.

### Avaliação da melhor segmentação

Para avaliar a melhor segmentação primeiramente utilizamos a função `adapted_rand_error` importada de `skimage.metrics`, essa função recebe como argumentos a segmentação que consideramos verdadeira e a segmentação que vai ser testada. O retorno são três floats que representam the nível de erros, o nível de precisão (o número de pares de pixeis que tem o mesmo 'rótulo' na segmentação verdadeira e na teste , dividido pelo número de label na imagem teste), e o nível de 'recall' (que é quase o mesmo que a precisão, mudando apenas que a divisão é pelo número de label na segmentação verdadeira). Essas métricas vão de 0.0 até 1.0 como valor máximo, avaliando a soma de todas as vinte imagens do padrão ouro o resultado foi 19.497342 para isodata, 19.999038 para canny e 19.986019 para flood_fill. Ict recebeu nota 20.0, mas como sabíamos que esse método é muito arisco preferimos não usá-lo para ter mais garantia de um maior número de resultados bons usando o método Canny + binary_fill_holes.
Após debates com o Squad, decidimos não usar `adapted_rand_error`, apesar dela ter bons resultados, e usar em seu lugar IOU propriamente dita, que é a divisão entre `np.bitwise_and` das duas imagens sobre `np.bitwire_or` das duas. Os resultados com esta nova métrica ficaram semelhantes as anteriores, mas com uma pequena baixa em cada uma das somas das três notas e isodata ficando na frente. Como já sabíamos os problemas que o método isodata tinha com algumas das fotografias e vendo o resultado que flood_fill apresentava, acabamos continuando a usar o nosso método canny como o escolhido. A conclusão que chegamos é que o ideal seria um número maior de padrões ouro para futuras aplicações, o que provavelmente ajudaria a ter um resultado IOU mais verdadeiro.

| Método | Soma IOU | Info |
| --- | --- | --- |
| isodata | 18.952114 | Nota boa com os as segmentações gold, mas problemática com algumas das outras fotografias | 
| canny | 18.748315 | O método baseado em Canny, apesar de ter a menor nota na soma dos padrões ouro, é a segmentação que apresenta resultados mais consistentes |
|flood_fill | 18.892653 | Flood fill após modificações resultou em grande parte com resultados muito bons, mas visivelmente inferiores aos canny |

## Métricas do padrão ouro

Utilizando a função `skimage.measure.regionprops_table` com conseguimos as métricas de axis_major_length, axis_minor_length e area de todas as imagens do padrão ouro e salvamos todas estas informações em um pandas dataframe e exportado para o disco em um arquivo csv. Também salvamos plotagens dessas informações nas segmentações que calculamos usando `skimage.measure.regionprops` e matplotlib, com ajuda de uma função do matplotlib que encontramos na documentação do skimage.

|  | names | axis_major_length | axis_minor_length | area
|--| --- | --- | --- | --- |
 0 | 01.jpg | 406.65144063150115 | 221.5456607581172  | 70437
 1 | 02.jpg | 462.0694091607201  | 191.74480388485068 | 68337
 2 | 03.jpg | 479.11449162993773 | 297.3196006969664  | 97358
 3 | 04.jpg | 451.2531870207678  | 252.10089228401532 | 87818
 4 | 05.jpg | 369.0600103522897  | 243.78849005386033 | 63955
 5 | 06.jpg | 255.5612465955241  | 109.31696307319861 | 21625
 6 | 07.jpg | 356.3398270370082  | 77.77759911689351  | 21391
 7 | 08.jpg | 293.6637806095889  | 194.9563636007646  | 39682
 8 | 09.jpg | 226.65809501121328 | 99.10667426234278  | 17132
 9 | 10.jpg | 194.8554014284213  | 94.79585294915192  | 14186
10 | 11.jpg | 175.41175684301987 | 122.88793813873762 | 15924
11 | 12.jpg | 431.7258848539172  | 159.01745613906687 | 48844
12 | 13.jpg | 161.62091707469622 | 111.06131134747368 | 13852
13 | 14.jpg | 503.16205023024935 | 195.5187967127831  | 76471
14 | 15.jpg | 297.71441805329926 | 53.94695981285079  | 12386
15 | 16.jpg | 549.7066034359743  | 188.3414915557074  | 79175
16 | 17.jpg | 470.94245213056234 | 197.00463339550947 | 71857
17 | 18.jpg | 364.5478371108046  | 215.66051021889794 | 61589
18 | 19.jpg | 400.0028996132361  | 126.28873906253686 | 37211
19 | 20.jpg | 718.120212793773   | 128.35792874832157 | 71787

## Aplicação em todas as outras imagens do dataset

Como fotografias de diferentes membros tinham resoluções, dimensões e qualidade diferentes para cada origem das fotos, executamos pré-processamentos personalizados para cada origem utilizando a ferramenta `convert` do ImageMagick. Primeiro executamos alguns testes rápidos para encontrar os números de pixels que deveriamos recortar todas as imagens e depois aplicado em todas com um for loop em bash.

Para cada uma das origens foi executado duas alterações em cada uma de suas imagens, onde todas as fotografias de cada origem estava em sua própria pasta. O comandos de pré-processamentos foram escritos em um for loop em bash, no terminal, que executava convert em cada uma das fotografias, sendo eles:

```bash
for image in *.jpg; do convert "$image" -resize x900 "$image"; done
for image in *.jpg; do convert "$image" -crop 0x0+60+0 "$image"; done

for image in *.jpg; do convert "$image" -crop 1400x1900+170+80 "$image"; done
for image in *.jpg; do convert "$image" -resize x900 "$image"; done
```

Os comandos com a flag `-crop` recortam cada uma das imagens para o valor especificado e os com a flag `-resize` reduzem o tamanho da imagem para 900px de altura. O valor do recorte foi baseado no tamanho da folha de papel a4 que todas as fotos tinham como fundo, mas cada origem apresentava mais ou menos de conteúdo nas fotos além do papel, por isso a diferença no recorte.

## Segmentação de todas as outras imagens

Após a escolha da melhor segmentação do padrão ouro ter sido definida como canny, segmentamos todas as outras imagens utilizando ela. Calculamos as métricas de cada uma e a salvamos em um csv. Todas as segmentações também foram exportadas lado a lado com a imagem original e montadas em um pdf com ajuda de `convert`.

## Streamlit

[[ adicionar relatorio streamlit ]] 

## Possíveis melhorias

Um ponto muito importante que pensamos que poderia ser de grande melhoria para o nosso resultado seria primeiramente iniciar com um grande número de padrões ouro, que ajudariam a solucionar o problema das métricas IOU de só das vinte ouro não condizerem com a realidade do resultado que as outras imagens apresentavam. Nas vinte ouro quase todos os métodos de segmentação da nossa classe funcionam muito bem, mas com as outras imagens a situação se mostrou um pouco aleatória, o que só reforça o fato que precisávamos de mais padrões ouro. Infelizmente a grande quantidade de chuvas no período do projeto impossibilitaram algum de nossos membros a coletar mais folhas.
O segundo ponto que poderíamos melhorar seria em conseguir executar as segmentações de forma assíncrona, segmentando várias fotografia paralelamente em vez de uma por vez como está agora. Isso possivelmente diminuiria drásticamente a quantidade de tempo necessário para segmentar cada grupo de imagens e facilitaria na realização de testes ao acelerar esse processo.
Com essas duas soluções também poderíamos modificar a nossa codebase para fazer todo o processo em apenas uma execução, onde automaticamente o nosso programa segmentaria os padrões ouro em todos os três métodos, mediria a melhor soma de IOU desses três e já aplicaria a melhor segmentação nas outras imagens. Atualmente precisamos executar as três distintas etapas separadamente para chegar ao resultado final, conseguimos desenvolver o que era pedido no projeto, mas poderíamos com essas mudanças evoluir o produto final.
