from .neighbors import neighbors_df
from pandas import DataFrame
import networkx as nx
import matplotlib.pyplot as plt

def plot_nx(df: DataFrame, norm: int) -> None:

    df_graph: DataFrame = neighbors_df(df)

    G = nx.from_pandas_edgelist(df_graph,source='source',target='target')

    df.set_index('tokens', inplace=True)
    # A pd.Series with the tf_mean of all nodes from G multiplied by norm
    nodes_sizes = df.loc[list(G.nodes)]['tf_mean']*norm
#
    plt.figure(figsize=(20, 20))
    colors: list[int] = [n for n in range(len(G.nodes()))]
    pos: dict = nx.spring_layout(G, k=0.35, seed=100)
    nx.draw_networkx(G, pos, node_size=nodes_sizes, cmap=plt.cm.RdYlGn, 
                     node_color=colors, edge_color='grey', 
                     font_size=12, style='--')
    plt.show()
