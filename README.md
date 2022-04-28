# Atlântico Computação Cognitiva

Trailhead scripts e atividades do Atlântico Bootcamp

## Módulo 1 - Processamento de Linguagem Natural

### ./trailhead/intro-NLP/
-	Introdução a NLP
-	NLP e ML

    `python main.py`

    Para plotar BoW CountVectorizer é necessário descomentar as linhas 82 e 83,
    do script main.py por padrão importance é a que vai ser plotada, mas
    alterando algum dos outros para True é possível plotar mais de um
    para BoW TfidfVectorizer funciona do mesmo jeito, mas estas linhas 86 e 87
    já estão descomentadas
    Word2Vec depende do dataset gigante e é necessário ter ele baixado ou
    montado no google drive (não testei esta opção pq executei tudo localmente)
    É necessário descomentar todas as linhas a partir da 94 até a 100 e
    escolher LSA, metrics ou confusion para descomentar

### ./trailhead/stanza-tests/
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

### ./trailhead/nltk-sentiment/
-   Análise de sentimentos com NLTK

    `python nltk_downloads.py` Caso você precise baixar os modelos

    `python nlp_test.py`
