import numpy as np

#Método para calcular a distância semântica entre as palavras
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) 
                      for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) 
                      for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

#Montagem do arquivo de treinamento contento a relação semântica entre as palavras
def get_word2vec_embeddings(vectors, data, generate_missing=False):
    embeddings = data['tokens'].apply(
            lambda x: get_average_word2vec(x, vectors, 
                      generate_missing=generate_missing))
    return list(embeddings)
