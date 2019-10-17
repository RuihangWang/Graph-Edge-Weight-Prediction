import random
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
import networkx as nx

def leave_out_n(G, n):
    """
    leave out n percentage of edges, but will keep at least one edge of each node
    :param G: input Graph
    :param n: n percentage edges will be removed, n ranges in(0,100]
    :return: Graph after leave out n percentage edges
    """
    G_n = G.copy()
    target = int(len(G.edges()) * (n / 100))
    all_edges=[]
    for (u, v) in G.edges():
        all_edges.append((u,v))
    slice = random.sample(all_edges,len(all_edges))
    slice = slice[:target]
    G_n.remove_edges_from(slice)

    return  G_n

def leave_out_edges(G, remove_edges_num):
    all_edges = []
    for (u, v) in G.edges():
        all_edges.append((u, v))
    remove_edges = random.sample(all_edges, remove_edges_num)
    return remove_edges


def predict_weight(WSN_method, G, G_n, u_v_edge=None):
    """

    :param WSN_method: class method, for example: Page_Rank
    :param G: original Graph including all edges
    :param G_n: G after removing edges
    :param u_v_edge: if None, we will predict all removed edges and return the RMSE and PCC values
                     if (u,v) is called, we will predict the weight of edge (u,v)
    :return: if None, we will predict all removed edges and return the RMSE and PCC values
            if (u,v) is called, we will predict the weight of edge (u,v)
    """

    total_w = []
    total_w_ = []

    if u_v_edge is not None:
        (u, v) = u_v_edge
        w = G[u][v]['weight']
        w_ = WSN_method.cal_w_(u, v)
        return w, w_

    for (u, v, w) in G.edges(data='weight'):
        if G_n.has_edge(u, v):
            continue
        w_ = WSN_method.cal_w_(u, v)
        total_w.append(w)
        total_w_.append(w_)
    RMSE = mean_squared_error(total_w, total_w_) ** 0.5
    PCC = pearsonr(total_w, total_w_)

    return RMSE, PCC[0]

def init_Graph(filename, path):
    """
    :param filename: Network.csv, path
    :return: nx.DiGraph()
    """
    G = nx.DiGraph()
    f = open(path + filename, "r")
    for l in f:
        ls = l.strip().split(",")
        if float(ls[2]) >= 0:
            w = 1
            p = float(ls[2])
            n = 0
        else:
            p = 0
            n = float(ls[2])
            w = 0
        G.add_edge(ls[0], ls[1], weight=float(ls[2]), signed_weight=w, positive=p, negative=n)

    f.close()

    return G

