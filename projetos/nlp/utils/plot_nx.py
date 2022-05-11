from .neighbors import five_n_neighbors_df
from .dataframe import NLPDataFrame
from pandas import DataFrame, Series
from typing import TypeAlias
import networkx as nx
import matplotlib.pyplot as plt

posDict: TypeAlias = dict[str, 'ndarray[float64]']

def build_graph(df_graph: DataFrame) -> nx.Graph:
    ''' Builds the networkx graph from a temporary neighbors DataFrame that has
    two columns 's' and 't', on s there are all the five words with the
    biggest tf_idf and t are all their neighbor words/close words '''
    return nx.from_pandas_edgelist(df_graph, source='s', target='t')

def get_pos(G: nx.Graph, k: float, s: int, it: int) -> posDict:
    ''' Returns a dict with str keys and ndarray[float64] as values that
    represent the position info for each node in G '''
    return nx.spring_layout(G, k=k, seed=s, iterations=it)

def get_node_sizes(five_largest: list[str], df: NLPDataFrame, 
                   G: nx.Graph, norm: int) -> Series:
    ''' Sets df's tokens column as index so we can use loc on all the words
    that are now nodes inside G and grab their tf_mean, it then loops through
    the five_largest list and multiplies them on the Series so that they are
    easily spotted on the plotted graph
    Arg:
        df (NLPDataFrame): the dataframe with all token words and nlp metrics
        G (nx.Graph): the networkx Graph with the list of nodes inside
        norm (int): Since the tf_mean is generally a really small number, we
        need to make them bigger by multipying with the normalizer
     '''
    nodes_sizes: Series 
    nodes_sizes = df.set_index('tokens').loc[list(G.nodes)]['tf_mean'] * norm
    for word in five_largest:
        # Enfasis on the five largest tf_idf_mean words
        nodes_sizes[word] = nodes_sizes[word] * 10
    return nodes_sizes

def plot_nx(df: NLPDataFrame, norm: int, k: float, 
            iterations: int, seed: int=1, savefig: bool=False) -> None:
    ''' Configures the figure to be plotted, builds a nx.Graph out of df, saves
    the figure if savefig is set to True and then shows the plot 
    Args:
        df (NLPDataFrame): Class with a has-a relationship with
        pandas.DataFrame. It's DataFrame contains metrics like tf, df, tf_idf,
        etc.
        norm (int): since the tf_mean on df is generally really small numbers,
        we use this norm to multiply them so that the nodes can look better in
        the plot
        k (float): small float from 0.1 and 0.5 that is used as a
        parameter in the nx.spring_layout function that generates a dictionary
        with the positions of nodes from the nx.Graph
        iterations (int): one of the keyword arguments from nx.spring_layout,
        numbers smaller than 50 make the nodes position look like a big ring
        around some of the big nodes, it is a little hard to see the source of
        the edges so bigger numbers are prefered (like 80, 100 and 200)
        seed (int): nx.spring_layout used a deterministic random number
        generation for the generation of positions, we can then set the seed
        number to make sure we have consistent results
        savefig (bool): if set to True the plotted figure is saved as a png
        file, or not saved if False '''

    # Configures the plot figure
    fig: plt.Figure; ax: plt.Axes
    fig, ax = plt.subplots(figsize=(24, 13.5))
    fig.tight_layout() # Makes the borders around the plot smaller

    # five largest words and it's neighbors
    five_largest: list[str]; df_graph: DataFrame
    five_largest, df_graph = five_n_neighbors_df(df)

    # Builds the nx.Graph and it's informations
    G: nx.Graph = build_graph(df_graph)
    colors: list[int] = [n for n in range(len(G.nodes()))]
    nodes_position: posDict = get_pos(G, k, seed, iterations)
    nodes_sizes: Series = get_node_sizes(five_largest, df, G, norm)

    # Draws the Graph
    nx.draw_networkx(G, nodes_position, node_size=nodes_sizes, 
                     node_color=colors, edge_color='grey', 
                     font_size=12, #font_weight='bold',
                     #style='--',
                     cmap=plt.cm.RdYlGn)

    if savefig:
        plt.savefig('exports/plot.png')
    plt.show()
