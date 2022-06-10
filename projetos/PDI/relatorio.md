# Projeto 2: Bootcamp Atlântico Academy - Computação Cognitiva - Squad III

Relatório sobre o segundo projeto para o bootcamp do Instituto Atlântico trilha Computação Cognitiva. Utilizando ferramentas como opencv-python ou scikit-image, bastante conhecidas em trabalhos de processsamento digital de imagens, este projeto tinha como objetivo segmentar um dataset de cem folhas de plantas que o grupo deveria colhetar e fotografar seguindo um padrão definido pelo próprio grupo.

O trabalho foi dividido em três etapas, sendo elas:
1. Definir um padrão ouro com vinte das cem folhas, suas segmentações verdadeiras e métricas de maior, menor comprimento e suas áreas.
2. Segmentar o padrão ouro utilizando três métodos diferentes de segmentação utilizando python e calcular qual desses métodos apresentavam o melhor resultado comparado as segmentações verdadeiras (Pintadas em um programa como o Krita).
3. Com a melhor segmentação, segmentar todas as outras imagens, calcular suas métricas e exportá-las em um csv, além de salvar a imagem original lado a lado com sua segmentação.

## Gold Standard

Fotografia de folhas variadas em cima de uma folha de papel tamanho A4 na posição retrato. Usando o programa de linha de comando `convert`, pertencente à suite de aplicações ImageMagick, todas as imagens de gold standard foram redimensionadas para 900px de altura. Escolhemos redimensionar as imagens para 900 pixeis de altura para uniformizar todas as fotografias em uma mesma resolução e também para conseguir trabalhar com elas de forma mais eficiente computacionalmente.
Inicialmente imaginamos que era obrigatório que as imagens usadas como gold standard deveriam ser segmentadas utilizando algum método do scikit-image ou opencv, e para isso foi escolhido o método limiar por isodata, que trazia bons resultados em grande quantidade de folhas, mas em algumas com muito brilho a segmentação ficava com muitos buracos que muitas vezes não podiam ser fechados usando funções prontas como `scipy.ndimage.binary_fill_holes`. Em um tutorial sobre como encontrar a melhor segmentação na documentação do scikit-image trazia a informação que para encontrar a melhor, era necessário comparar com a ouro/segmentação verdadeira. Surgiu aí a dúvida sobre se isodata serviria como ouro, sabendo que algumas segmentações não estavam perfeitas, e que ela na verdade deveria ser comparada com outra segmentação superior. No mesmo tutorial sobre encontrar a melhor segmentação incluía uma frase sobre ser comum o padrão ouro ser feito manualmente, isso respondeu a principal dúvida sobre o padrão ouro e foi o que produzimos em seguida. Usando o software krita segmentamos manualmente com ajuda de uma tablet wacom todos os padrões ouro.
Com o padrão ouro em mãos e a segmentação por isodata já feita, segmentamos as folhas do padrão ouro mais algumas vezes com outros métodos, e os finais escolhidos foram os métodos canny, flood_fill e o já mencionado limiar por isodata.

## Segmentação dos Padrões Ouro

Antes da proposta do projeto já tinhamos iniciado os trabalhos em uma classe com métodos de segmetação utilizando scikit-image. Após receber o projeto, continuamos trabalhando nesta classe com refatoração do código e acrescentando novos métodos e melhor organização no módulo segmentation.py. Essa classe chamada Segment recebe obrigatoriamente uma string com o path para uma imagem, ela automaticamente carrega essa imagem utilizando a função `skimage.io.imread` e também guarda uma versão preto e branco da imagem original e uma versão com gaussian blur aplicado na versão preto e branca, caso seja necessária. A classe continua com a definição de vários métodos de segmentação, como os já citados limiar por isodata, canny, flood_fill, mas também vários outros tipos de limar, ict, entre outras.

### Métodos de segmentação

1. *****Isodata********


    A classe Segment também inclui um método que utiliza `skimage.filters.thresholding.try_all_threshold` que testa vários tipos de limiares para análise subjetiva de qual é o melhor. Nas imagens utilizadas como padrão ouro o melhor limiar foi o isodata. Isodata resultou em segmentações geralmente bem limpas, mas imperfeitas, enquanto muitos dos outros tipos de limiar faziam sombras nas fotografias ficarem completamente pretas, e o método limiar local em especial que nas nossas fotografias resultava em imagens totalmente ou quase totalmente pretas.

2. **Canny**

    Canny é um método de detecção de bordas que trás resultados geralmente muito bons e detalhados. Nessas bordas detectadas com o algoritmo canny foi aplicado a função `scipy.ndimage.binary_fill_holes` que preenche totalmente formas fechadas, resultando em uma segmentação com alto nível de precisão na maioria das imagens. O grande problema desse método foi com folhas muito claras, com partes brancas ou folhas grande demais, que passavam um pouco do tamanho do papel a4, o que resultava nas bordas detectadas nesses casos não serem fechadas e binary_fill_holes não poder preencher elas.

3. **Flood fill**

    Sem dúvidas este foi um dos métodos mais divertidos de escrever. A função flood_fill do skimage precisa de uma seed onde ela é aplicada, essa seed é uma tuple de dois int que representa o pixel no pd.ndarray (row, column), que é a imagem, que flood_fill vai começar a atuar. Como cada imagem tinha uma folha de espécie diferente, tamanhos diferentes, e nenhuma perfeitamente centralizada na folha de papel a4, era impossível escolher um ponto fixo para ser a seed. Para resolver este problema, após buscas, escrevemos um pequeno staticmethod que recebe a imagem como argumento, transforma essa uma cópia dessa imagem, um pd.ndarray, em um array de uma única dimensão com o método flatten do próprio array e desse array encontramos o menor valor com a função min(), em seguida utilizando np.where procuramos o index desse menor valor no ndarray original da imagem e retornamos uma tuple de dois int que é o index (row, column) do pixel mais escuro na imagem. Com essa seed em mãos aplicamos flood_fill com tolenrancia padrão de 100, o que se provou nas 20 imagens ouro suficiente para pintar toda a folha de branco sem tocar na cor do fundo. Com a folha pintada com o valor 255, extraímos essa informação do array original utilizando uma expressão booleana de igualdade e para garantir que essa imagem binária não tem buracos na folha utilizamos `scipy.ndimage.binary_fill_holes` nela.

4. **Outros testes**

    Também realizamos testes com os métodos de segmentação Iterative Cluster Threshold, Felzenszwalb, Chanvese e outros. Visualmente todos esses apresentavam resultados muito interessantes, ict em espacial que recebeu notas altíssimas com sua precisão avaliada. O grande problema com ict (iterative cluster threshold) é que esse método se mostrou um pouco arisco e com frequentes falhas ao segmentar bem todas as folhas, algumas muito pequenas sem serem segmentadas completamente ou maiores que se unia ao fundo branco em pontas aleatórias. Muito tempo foi utilizado tentando encontrar um balanceamento nos dois principais argumentos da função usada para gerar ict. Chanvese chamou atenção com seus resultados esteticamente interessantes, mas completamente inúteis para as fotografias que tinhamos em mãos para serem o padrão ouro.

### Avaliação da melhor segmentação

Para avaliar a melhor segmentação utilizamos a função `adapted_rand_error` importada de `skimage.metrics`, essa função recebe como argumentos a segmentação verdadeira que consideramos verdadeira e a segmentação que vai ser testada. O retorno são três floats que representam the nível de erros, o nível de precisão (o número de pares de pixeis que tem o mesmo 'rótulo' na segmentação verdadeira e na teste , dividido pelo número de label na imagem teste), e o nível de 'recall' (que é quase o mesmo que a precisão, mudando apenas que a divisão é pelo número de label na segmentação verdadeira). Essas métricas vão de 0.0 até 1.0 como valor máximo, avaliando a soma de todas as vinte imagens do padrão ouro o resultado foi 19.497342 para isodata, 19.999038 para canny e 19.986019 para flood_fill. Ict recebeu nota 20.0, mas como sabíamos que esse método é muito arisco preferimos não usá-lo para ter mais garantia de um maior número de resultados bons usando o método Canny + binary_fill_holes.

## Métricas do padrão ouro

Utilizando a função `skimage.measure.regionprops_table` com conseguimos as métricas de axis_major_length, axis_minor_length e area de todas as imagens do padrão ouro e salvamos todas estas informações em um pandas dataframe e salvo em disco como um csv. Também salvamos plotagens dessas informações nas segmentações que calculamos usando `skimage.measure.regionprops` e matplotlib, com ajuda de uma função do matplotlib que encontramos na documentação do skimage.

```
    axis_major_length   axis_minor_length   area
0   161.62091707469622  111.06131134747368  13852
1   503.16205023024935  195.5187967127831   76471
2   718.120212793773    128.35792874832157  71787
3   431.7258848539172   159.01745613906687  48844
4   470.94245213056234  197.00463339550947  71857
5   549.7066034359743   188.3414915557074   79175
6   462.0694091607201   191.74480388485068  68337
7   369.0600103522897   243.78849005386033  63955
8   451.2531870207678   252.10089228401532  87818
9   194.8554014284213   94.79585294915192   14186
10  293.6637806095889   194.9563636007646   39682
11  406.65144063150115  221.5456607581172   70437
12  175.41175684301987  122.88793813873762  15924
13  356.3398270370082   77.77759911689351   21391
14  255.5612465955241   109.31696307319861  21625
15  400.0028996132361   126.28873906253686  37211
16  297.71441805329926  53.94695981285079   12386
17  479.11449162993773  297.3196006969664   97358
18  364.5478371108046   215.66051021889794  61589
19  226.65809501121328  99.10667426234278   17132
```

## Aplicação em todas as outras imagens do dataset

Para as outras imagens, como elas eram de resoluções, dimensões e qualidade diferentes para cada origem das fotos, executamos pré-processamentos personalizados para cada origem utilizando a ferramenta `convert` do ImageMagick. Primeiro executamos alguns testes rápidos para encontrar os números de pixels que deveriamos recortar todas as imagens.

Para cada uma das duas origens foi executado duas alterações em cada uma de suas imagens, onde todas as fotografias de cada origem estava em sua própria pasta. O comandos de pré-processamentos foram escritos em um for loop em bash,no terminal, que executava convert em cada uma das fotografias, sendo eles:

```
for image in *.jpg; do convert "$image" -resize x900 "$image"; done
for image in *.jpg; do convert "$image" -crop 0x0+60+0 "$image"; done

for image in *.jpg; do convert "$image" -crop 1400x1900+170+80 "$image"; done
for image in *.jpg; do convert "$image" -resize x900 "$image"; done
```

Os comandos com a flag `-crop` recortam cada uma das imagens para o valor especificado e os com a flag `-resize` reduzem o tamanho da imagem para 900px de altura. O valor do recorte foi baseado no tamanho da folha de papel a4 que todas as fotos tinham como fundo, mas cada origem apresentava mais ou menos de conteúdo nas fotos além do papel, por isso a diferença no recorte.

## Segmentação de todas as outras imagens

Após a escolha da melhor segmentação do padrão ouro ter sido definida como canny, segmentamos todas as outras imagens utilizando ela. Calculamos as métricas de cada uma e a salvamos em um csv. Todas as segmentações também foram exportadas lado a lado com a imagem original e montadas em um pdf com ajuda de `convert`.

## Possíveis melhorias

Infelizmente o códigos escritos para este projeto, do jeito que estão, não fazem todas as três etapas em apenas uma execução, requerendo manualmente editar main.py para executar cada uma das três. Em um espaço de mais um sprint possivelmente poderíamos implementar esta melhora para transformar o projeto em um produto/programa mais completo.
