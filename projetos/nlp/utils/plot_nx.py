from .neighbors import neighbors_df
from .dataframe import NLPDataFrame
from numpy import ndarray, float64
from pandas import DataFrame, Series
from typing import TypeAlias
import networkx as nx
import matplotlib.pyplot as plt

#posDict: TypeAlias = dict[str, ndarray[float]]
posDict: TypeAlias = dict[str, ndarray]

def get_node_sizes(df: NLPDataFrame, G: nx.Graph, norm: int) -> Series:
    ''' Sets df's tokens column as index so we can use loc on all the words
    that are now nodes inside G and grab their tf_mean, it then returns this
    pd.Series multiplied by the norm
    Arg:
        df (NLPDataFrame): the dataframe with all token words and nlp metrics
        G (nx.Graph): the networkx Graph with the list of nodes inside
        norm (int): Since the tf_mean is generally a really small number, we
        need to make them bigger by multipying with the normalizer
     '''
    return df.set_index('tokens').loc[list(G.nodes)]['tf_mean'] * norm

def build_graph(df: NLPDataFrame) -> nx.Graph:
    ''' Builds the networkx graph from a temporary neighbors DataFrame that has
    two columns 's' and 't', on s there are all the five words with the
    biggest tf_idf and t are all their neighbor words/close words '''
    return nx.from_pandas_edgelist(neighbors_df(df), source='s', target='t')

def get_pos(G: nx.Graph, k: float, s: int, it: int) -> posDict:
    ''' Returns a dict with the position info for each node in G '''
    return nx.spring_layout(G, k=k, seed=s, iterations=it)

def plot_nx(df: NLPDataFrame, 
            norm: int, 
            spacing: float, 
            iterations: int,
            seed=1, 
            savefig=False) -> None:

    # Configures the plot figure
    fig: plt.Figure; ax: plt.Axes
    fig, ax = plt.subplots(figsize=(24, 13.5))
    fig.tight_layout() # Makes the borders around the plot smaller

    # Builds the nx.Graph and it's informations
    G: nx.Graph = build_graph(df)
    colors: list[int] = [n for n in range(len(G.nodes()))]
    pos: posDict = get_pos(G, spacing, seed, iterations)

    # Draws the Graph
    nx.draw_networkx(G, pos,
                     node_size=get_node_sizes(df, G, norm), 
                     node_color=colors, edge_color='grey', 
                     font_size=12, font_weight='bold',
                     #style='--',
                     cmap=plt.cm.RdYlGn)

    if savefig:
        plt.savefig('exports/plot.png')
    plt.show()
