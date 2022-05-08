from pandas import DataFrame, concat

def get_neighbors(flat_lemmas: list[str], word: str) -> list[str]:
    ''' Returns a list of neighbors of word in tokens '''
    flat_lemmas_len: int = len(flat_lemmas)
    neighbors: list[str] = []
    start: int = 0
    while True:
        try:
            index: int = flat_lemmas.index(word, start)
            # Only tries to append index+1 if it's smaller than
            # flat_lemmas_len-1
            if index + 1 < flat_lemmas_len-1:
                neighbors.append(flat_lemmas[index+1])
            neighbors.append(flat_lemmas[index-1])
            start = (index + 1) 
        except ValueError:
            break
    return neighbors

def neighbors_df(df: DataFrame) -> DataFrame:

    five_largest = df.nlargest(5, 'tf_idf_mean')['tokens'].tolist()

    df_graph = DataFrame()
    for word in five_largest:
        n = get_neighbors(df.flat_lemmas, word)
        df_tmp = DataFrame({'t': n, 's': [word]*len(n)})
        df_graph = concat([df_graph, df_tmp], ignore_index=True)

    return df_graph
