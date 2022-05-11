from pandas import DataFrame, concat # type: ignore
from .dataframe import NLPDataFrame

def get_neighbors(flat_lemmas: list[str], word: str) -> list[str]:
    ''' Returns a list of all the neighbors of word in flat_lemmas '''
    flat_lemmas_len: int = len(flat_lemmas)
    neighbors: list[str] = []
    start: int = 0
    while True:
        try:
            index: int = flat_lemmas.index(word, start)
            # Only tries to append index+2 if it's smaller than
            # flat_lemmas_len-1
            for i in [1, 2]:
                if index > (i - 1):
                    neighbors.append(flat_lemmas[index-i])
                if index + i <= flat_lemmas_len-1:
                    neighbors.append(flat_lemmas[index+i])
            #only one neighbor from each side
            #if index + 1 < flat_lemmas_len-1:
            #    neighbors.append(flat_lemmas[index+1])
            #neighbors.append(flat_lemmas[index-1])
            start = (index + 1) 
        except ValueError:
            break
    return neighbors

def five_n_neighbors_df(df: NLPDataFrame) -> tuple[list[str], DataFrame]:
    ''' Returns the list with the five words with the highest tf_idf_mean and a
    dataframe with a column called 't' that are all the neighbors and a 's'
    column that is one of the words in df with the largest tf_idf_mean. All
    items in the 't' column are in a later stage used as target and the ones in
    's' as source when creating a networkx.Graph using
    networkx.from_pandas_edgelist
    '''

    five_largest: list[str] = df.nlargest(5, 'tf_idf_mean')['tokens'].tolist()

    df_graph = DataFrame()
    for word in five_largest:
        n: list[str] = get_neighbors(df.flat_lemmas, word)
        df_tmp = DataFrame({'t': n, 's': [word]*len(n)})
        df_graph = concat([df_graph, df_tmp], ignore_index=True)

    # Returns a DataFrame after removing rows that have the same word in 't'
    # and 's' to avoid a node linking to itself in the graph
    df_graph = df_graph[df_graph['t'] != df_graph['s']]
    return five_largest, df_graph
