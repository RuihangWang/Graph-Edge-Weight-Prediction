import networkx as nx
from scipy.stats.stats import pearsonr

def signed_hits(G, G_n, h, a):

    G_hits = G_n.copy()
    total_w = []
    total_w_ = []
    w_in = {}
    w_out = {}
    Hits_total_out = {}
    Hits_total_in = {}

    for (u, w) in G_hits.nodes(data='weight'):
        w_in_v = 0
        w_out_v = 0
        Hits_total_out_v = 0
        Hits_total_in_v = 0
        for edge in G_hits.out_edges(u, data='weight'):
            w_out_v += edge[2] * h[edge[0]]
            Hits_total_out_v += h[edge[0]]
        for edge in G_hits.in_edges(u, data='weight'):
            w_in_v += edge[2] * a[edge[1]]
            Hits_total_in_v += a[edge[1]]
        w_in[u] = w_in_v
        w_out[u] = w_out_v
        Hits_total_in[u] = Hits_total_in_v
        Hits_total_out[u] = Hits_total_out_v
    nx.set_node_attributes(G_hits, w_in, 'w_in')
    nx.set_node_attributes(G_hits, w_out, 'w_out')
    nx.set_node_attributes(G_hits, Hits_total_in, 'Hits_total_in')
    nx.set_node_attributes(G_hits, Hits_total_out, 'Hits_total_out')

    RMSE = 0
    iter = 0

    for (u, v, w) in G.edges(data='weight'):
        if G_hits.has_edge(u, v):
            continue
        w_ = 0
        if G_hits.node[u]['Hits_total_out'] !=0 :
            w_ += G_hits.node[u]['w_out']/G_hits.node[u]['Hits_total_out']
        if G_hits.node[v]['Hits_total_in'] != 0 :
            w_ += G_hits.node[v]['w_in']/G_hits.node[v]['Hits_total_in']
        w_ /= 2
        iter += 1
        RMSE += (w_ - w) ** 2
        total_w.append(w)
        total_w_.append(w_)
    RMSE /= iter
    RMSE = RMSE ** 0.5
    PCC = pearsonr(total_w, total_w_)

    return RMSE, PCC[0]


