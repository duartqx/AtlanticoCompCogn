from pandas import DataFrame, concat
from .dataframe import NLPDataFrame

def get_neighbors(flat_lemmas: list[str], word: str) -> list[str]:
    ''' Returns a list of all the neighbors of word in flat_lemmas '''
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

def neighbors_df(df: NLPDataFrame) -> DataFrame:
    ''' Returns a dataframe with a column called 't' that are all the neighbors
    and a 's' column that is one of the words in df with the largest
    tf_idf_mean. All items in the 't' column are in a later stage used as
    target and the ones in 's' as source when creating a networkx.Graph using
    networkx.from_pandas_edgelist
    '''

    five_largest: list[str] = df.nlargest(5, 'tf_idf_mean')['tokens'].tolist()

    df_graph = DataFrame()
    for word in five_largest:
        n: list[str] = get_neighbors(df.flat_lemmas, word)
        df_tmp = DataFrame({'t': n, 's': [word]*len(n)})
        df_graph = concat([df_graph, df_tmp], ignore_index=True)

    return df_graph
