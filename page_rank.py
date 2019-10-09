"""
Page Rank
"""
import networkx as nx
def pagerank_predict_weight(G, G_n, PR):
    RMSE = 0
    iter = 0
    target = len(G.edges()) - len(G_n.edges())
    G_PR = G_n.copy()

    w_in = {}
    w_out ={}
    PR_total_out = {}
    PR_total_in = {}
    for (u,w) in G_PR.nodes(data='weight'):
        w_in_v = 0
        w_out_v = 0
        PR_total_out_v = 0
        PR_total_in_v = 0
        for edge in G_n.out_edges(u, data='weight'):
            w_out_v += edge[2] * PR[edge[1]]
            PR_total_out_v += PR[edge[1]]
        for edge in G_n.in_edges(u,data='weight'):
            w_in_v += edge[2] * PR[edge[0]]
            PR_total_in_v += PR[edge[0]]
        w_in[u] = w_in_v
        w_out[u] = w_out_v
        PR_total_in[u] = PR_total_in_v
        PR_total_out[u] = PR_total_out_v
    nx.set_node_attributes(G_PR, w_in, 'w_in')
    nx.set_node_attributes(G_PR, w_out, 'w_out')
    nx.set_node_attributes(G_PR, PR_total_in, 'PR_total_in')
    nx.set_node_attributes(G_PR, PR_total_out, 'PR_total_out')

    for (u, v, w) in G.edges(data='weight'):
        if G_n.has_edge(u, v):
            continue
        w_ = G_PR.node[u]['w_out'] + G_PR.node[v]['w_in']
        PR_total = G_PR.node[u]['PR_total_out'] + G_PR.node[v]['PR_total_in']
        if PR_total != 0 :
            w_ /= PR_total

        iter += 1
        RMSE += (w_ - w) ** 2
    RMSE /= iter
    RMSE = RMSE ** 0.5
    return RMSE




