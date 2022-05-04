# Atlântico Computação Cognitiva

Trailhead scripts e atividades do Atlântico Bootcamp

## Módulo 1 - Processamento de Linguagem Natural

### ./trailhead/intro-NLP/
-	Introdução a NLP
-	NLP e ML

    `python main.py`

    Para plotar BoW CountVectorizer é necessário modificar a variavel
    main_kwargs para dict(tfidf=False, count=True) ou dict(count=True) ou
    passar diretamente count=True ao chamar a função main
    do script main.py
    Por padrão importance é a que vai ser plotada, mas
    alterando what2plot é possível configurar o que plotar em vez de imp
    BoW TfidfVectorizer funciona do mesmo jeito, mas este já é o método padrão
    Word2Vec depende do dataset gigante e é necessário ter ele baixado ou
    montado no google drive (não testei esta opção pq executei tudo localmente)
    É necessário configurar tfidf e count para False para Word2Vec ser
    executada pela função main

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
