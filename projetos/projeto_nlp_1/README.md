### Atlântico Bootcamp

# NLP

### VISÃO GERAL
Uma empresa contratante deseja estabelecer termos de maior relevância em um documento específico. Neste caso, considere o histórico de exames, consultas e procedimentos realizados por um paciente. Um sistema deve ser desenvolvido para que o médico possa ter uma visão geral do histórico do paciente sem a necessidade de analisar documento por documento. Com base nesta importância, vamos desenvolver uma etapa deste sistema. Tokenizar um texto, realizar remoção de stopwords, aplicar o processo de lematização e fazer uma análise quantitativa e visual subjetiva deste.

### OBJETIVOS
1. Carregar o conjunto de documentos em PDF e armazená-los em alguma estrutura de dados.
2. Realizar o pré-processamento destes ( tokenização e remoção de stop words, deixar todos os
caracteres minúsculos...).
3. Lematização com a Lib stanza
4. Implementar para determinar as seguintes informações dos resultados obtidos em 3 :

	1. Term Frequency (TF):
        TF = *qtd de ocorrência do termo em um texto / quantidade total de palavras do texto*
	2. Document Frequency (DF)
        DF = *qtd de ocorrência do termo em um conjunto de documentos*
	3. Inverse Document Frequency (IDF)
        IDF = *log(qtd de documentos / (DF + 1))*
	4. TF-IDF
        TF-IDF = *IDF \* TF*
	5. Lista de strings com proximidade até 2 dos 5 termos de maior TF-IDF. Essas strings devem ser acompanhadas de seu valor de TF. Exemplo: Suponha que a lista dos 5 termos de maior TF-IDF é [ casa, carro, comida, cachorro, gato]. Carro em um uma frase pode ter pneu e banco com as palavras mais próximas. Em outra parte do texto, carro pode ter volante e cinto, como as palavras mais próximas. Neste caso, para o termo carro, as strings [pneu,banco,volante,cinto] são as que devem ser armazenadas para análise.
	
5. Gerar um arquivo csv que possui todas as palavras de todos os documentos na primeira coluna, em que cada linha é um token. Para cada token, informe nas colunas vizinhas as informações determinadas no objetivo 4.1 até 4.4. 

6. Gerar nuvem de palavras para análise visual tal como exemplo abaixo. Cada ponto central será um dos 5 termos de maior TF-IDF. As conexões são as palavras próximas obtidas em 4.5. O tamanho do círculo da palavra é baseado no TF dela. O maior círculo que conecta o termo central será normalizado para palavras de maior TF do conjunto.

![graphic](https://www.kaggleusercontent.com/kf/7051625/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..0t_YoRKJ-5bXSZMzMNSlBA.9Cw6tMliJL4M2QxN7LicJqwSlhkIo0kdUv4IJ4vrXzaTeFWtRVoZcTA2mGTPeepEc4CcgDbZFcpd8mXM8semIIjye_CpuxtqrGFSFkKcgNdZ8NCDc8Ik5tibo132YyOrjv6ZMvP8j56ugpuJhFCOWBe5fjNBZbGWMbiOyULim7qyeQ6OheVB_KX5hk8sD4bGk7ZXHJ_HBhrfck_V3NYC4czt-4fG8MC-Mh_Uo5eIZUgj-sqrI0yNR8H8evZ2vuiEA4zaJTQjRH3Nt2CW59_WlXQUSNmDBCAAurW0gHq8Vj3VhqNrgXPgade3hsJD4GNEH5tWY7EWKr7Y8Vi8ogsAquUTfVhVIm6bCjK-QGSU-GAg4xplzF4RVpCJvDjqhCRCzU2SC6C2YszbNmA6cq6GEXPi0wyvhT9wmD78wTMhFgp2jyevzwRrmjjfTKYdUjgCcZE1njhj0ed7R-UjOYCPk9uWXXzyLPCmccTcT8WXTAaUUVFEFvMAscihArT6DO4JtYa40IM-i2wuoPGd3Cv-owoQTrxxJv32xZbp2wfMzBbD3oETwWGlp6DCwhUwV_rA-rVFXHm6i6P4n0rzOKEmL2sQbyQIucHXcPeGVEnknwjq18Ju4IvTJMlWknkREzHvM1vZf-2lLYuQXjuEoWx1Jg04XY4KIuXktgR5yYkSsJw1XIZYUW-sTqwi4ihDKNyZ.oQ-J-C15nDSGGEV2ubCViA/__results___files/__results___21_1.png "graphic")

### Tópicos de Auxílio
- Informações sobre as métricas utilizadas: \
https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-worlddataset-796d339a4089
- Atividade determinação da nuvem de palavras: \
https://www.kaggle.com/arthurtok/ghastly-network-and-d3-js-force-directed-graphs \
http://andrewtrick.com/stormlight_network.html
