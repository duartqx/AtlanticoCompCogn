## Gold Standard

Fotografia de folhas variadas em cima de uma folha de papel tamanho A4 na posição retrato. Usando o programa de linha de comando `convert`, pertencente à suite de aplicações ImageMagick, todas as imagens de gold standard foram redimensionadas para 900px de altura. 
Inicialmente imaginamos que era obrigatório que as imagens usadas como gold standard deveriam ser segmentadas utilizando algum método do scikit-image ou opencv, e para isso foi escolhido o método limiar por isodata, que trazia bons resultados em grande quantidade de folhas, mas em algumas com muito brilho a segmentação ficava com muitos buracos que muitas vezes não podiam ser fechados usando funções prontas como scipy.ndimage.binary_fill_holes. Em um tutorial sobre como encontrar a melhor segmentação na documentação do scikit-image trazia a informação que para encontrar a melhor, era necessário comparar com a ouro/segmentação verdadeira. Surgiu aí a dúvida sobre se isodata serviria como ouro, sabendo que algumas segmentações não estavam perfeitas, e que ela na verdade deveria ser comparada com outra segmentação superior. No mesmo tutorial sobre encontrar a melhor segmentação incluía uma frase sobre ser comum o padrão ouro ser feito manualmente, isso respondeu a principal dúvida sobre o padrão ouro e foi o que produzimos em seguida. Usando o software krita segmentamos manualmente com ajuda de uma tablet wacom todos os padrões ouro.
Com o padrão ouro em mãos e a segmentação por isodata já feita, segmentamos as folhas do padrão ouro mais algumas vezes com outros métodos, e os finais escolhidos foram os métodos canny, flood_fill e o já mencionado limiar por isodata.

## Segmentação dos Padrões Ouro

Antes da proposta do projeto já tinhamos iniciado os trabalhos em uma classe com métodos de segmetação utilizando scikit-image. Após receber o projeto, continuamos trabalhando nesta classe com refatoração do código e acrescentando novos métodos e melhor organização no módulo segmentation.py. Essa classe chamada Segment recebe obrigatoriamente uma string com o path para uma imagem, ela automaticamente carrega essa imagem utilizando a função skimage.io.imread e também guarda uma versão preto e branco da imagem original e uma versão com gaussian blur aplicado na versão preto e branca, caso seja necessária. A classe continua com a definição de vários métodos de segmentação, como os já citados limiar por isodata, canny, flood_fill, mas também vários outros tipos de limar, ict, entre outras.

### Segmentações escolhidas como as três exigidas no projeto

1. Isodata
    Essa classe também inclui um método que utiliza skimage.filters.thresholding.try_all_threshold que testa vários tipos de limiares para encontrar de forma subjetiva o melhor. Nas imagens utilizadas como padrão ouro o melhor limiar foi o isodata, pois os outros era muito comum sombras nas fotografias ficarem completamente pretas, enquanto isodata era geralmente muito limpo, mas ainda imperfeito.

2. Canny
    Canny é um método de detecção de bordas que trás resultados geralmente muito bons. Nessas bordas detectadas com o algoritmo canny foi aplicado a função scipy.ndimage.binary_fill_holes que preenche totalmente formas fechadas, resultando em uma segmentação com alto nível de precisão na maioria das imagens. O grande problema desse método foi com folhas muito claras, com partes brancas ou folhas grande demais, que passavam um pouco do tamanho do papel a4, o que resultava nas bordas detectadas nesses casos não serem fechadas e binary_fill_holes não podendo atuar nelas.

3. Flood fill
    Sem dúvidas este foi um dos métodos mais divertidos de escrever. A função flood_fill do skimage precisa de uma seed onde ela é aplicada, essa seed é uma tuple de dois int que representa o pixel no pd.ndarray (row, column), que é a imagem, que flood_fill vai começar a atuar. Como cada imagem tinha uma folha de espécie diferente, tamanhos diferentes, e nenhuma perfeitamente centralizada na folha de papel a4, era impossível escolher um ponto fixo para ser a seed. Para resolver este problema, após buscas, escrevemos um pequeno staticmethod que recebe a imagem como argumento, transforma essa uma cópia dessa imagem, um pd.ndarray, em um array de uma única dimensão com o método flatten do próprio array e desse array encontramos o menor valor com a função min(), em seguida utilizando np.where procuramos o index desse menor valor no ndarray original da imagem e retornamos uma tuple de dois int que é o index (row, column) do pixel mais escuro na imagem. Com essa seed em mãos aplicamos flood_fill com tolenrancia padrão de 100, o que se provou nas 20 imagens ouro suficiente para pintar toda a folha de branco sem tocar na cor do fundo. Com a folha pintada com o valor 255, extraímos essa informação do array original utilizando uma expressão booleana de igualdade e para garantir que essa imagem binária não tem buracos na folha utilizamos scipy.ndimage.binary_fill_holes nela.

Alguns outros métodos foram testados como Iterative Cluster Threshold, Felzenszwalb, Chanvese e outros. Visualmente todos esses apresentavam resultados muito interessantes, ict em espacial que recebeu notas altíssimas com sua precisão avaliada. O grande problema com ict (iterative cluster threshold) é que esse método é um pouco arisco e frequentemente falhava em segmentar bem as folhas, com algumas muito pequenas sem serem segmentadas completamente, e muito tempo foi utilizado tentando encontrar um balanceamento em seus dois principais argumentos. Chanvese chamou atenção com seus resultados esteticamente interessantes, mas completamente inúteis para as fotografias que tinhamos em mãos para serem o padrão ouro.

## Medindo melhor segmentação