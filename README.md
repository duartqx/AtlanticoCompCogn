# Atlântico Computação Cognitiva

![atlantico](https://github.com/duartqx/images/blob/main/atlantico.jpg?raw=true 'Atlantico Bootcamp')

Trailhead scripts e atividades do Atlântico Bootcamp

### Módulo 1 - Processamento de Linguagem Natural

#### ./trailhead/NLP/intro-NLP/
-	Introdução a NLP
-	NLP e ML

    `python main.py`

    Para plotar BoW CountVectorizer é necessário modificar a invocação da função main com CountVectorizer em vez de TfidfVectorizer que está como padrão. 
    Modifique também o dicionário what2plot para o método de plotagem que você quer.
    BoW TfidfVectorizer funciona do mesmo jeito, mas este já é o método padrão
    No caso de Word2Vec, remove TfidfVectorizer da invocação de main que Word2Vec será usado no lugar. Word2Vec depende de um dataset gigante e é necessário ter ele baixado ou montado no google drive (não testei esta opção pq executei tudo localmente)

#### ./trailhead/NLP/stanza-tests/
-	Tokenização e segmentação de sentenças

    `python stanza_tokenization_segmentation.py`

    Como baixei os modelos em uma pasta alternativa da padrão (que ficava na
    minha home), tive que indicar através da variável directory o local que
    salvei os modelos, caso não seja necessário no seu caso, comente
    'directory' e remove o das chamadas a print_pipeline

-   Lematização

    `python stanza_lema.py`

    Lembre-se de remover dir='../../venv/models' da chamada da classe Pipeline
    caso seu modelo tenha sido salvo em alguma pasta padrão, ou substitua pela
    pasta que você usou

#### ./trailhead/NLP/nltk-sentiment/
-   Análise de sentimentos com NLTK

    `python nltk_downloads.py` Caso você precise baixar os modelos

    `python nlp_test.py`

### Módulo 2 - Visão Computacional

#### ./trailhead/VC/OpenCV
-   Tutorial Opencv 

    `python main.py`

    main.py import a classe ImageEditor e executa alguns testes dos métodos desta classe que editam a imagem ponte.jpg presente na pasta resources e salva todas as edições na pasta exports

<p align="center">
    <img width="250" src="https://github.com/duartqx/images/blob/main/AtlanticoLogo.png?raw=true">
</p>
