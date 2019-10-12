"""
Page Rank
"""
import networkx as nx
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error

def pagerank_PR_graph(G_n, PR):
    G_PR = G_n.copy()

    w_in = {}
    w_out = {}
    PR_total_out = {}
    PR_total_in = {}

    # 对G_PR图的所有节点n计算out edge和in edge的边权重在PageRank值上的加权平均
    for (u, w) in G_PR.nodes(data='weight'):
        w_in_v = 0
        w_out_v = 0
        PR_total_out_v = 0
        PR_total_in_v = 0
        for edge in G_n.out_edges(u, data='weight'):
            w_out_v += edge[2] * PR[edge[1]]
            PR_total_out_v += PR[edge[1]]
        for edge in G_n.in_edges(u, data='weight'):
            w_in_v += edge[2] * PR[edge[1]]
            PR_total_in_v += PR[edge[1]]
        w_in[u] = w_in_v
        w_out[u] = w_out_v
        PR_total_in[u] = PR_total_in_v
        PR_total_out[u] = PR_total_out_v
    nx.set_node_attributes(G_PR, w_in, 'w_in')
    nx.set_node_attributes(G_PR, w_out, 'w_out')
    nx.set_node_attributes(G_PR, PR_total_in, 'PR_total_in')
    nx.set_node_attributes(G_PR, PR_total_out, 'PR_total_out')

    return G_PR

def cal_w_(u, v, G_PR):
    # 用源节点u的out edge和目标节点v的in edge的
    # edge weight 在 PageRank值上的加权平均
    # 作为预测值
    w_ = G_PR.node[u]['w_out'] + G_PR.node[v]['w_in']
    PR_total = G_PR.node[u]['PR_total_out'] + G_PR.node[v]['PR_total_in']
    if PR_total != 0:
        w_ /= PR_total
    return w_

def pagerank_predict_weight(G, G_PR, u_v_edge=None):

    total_w = []
    total_w_ = []

    if u_v_edge is not None:
        (u,v) = u_v_edge
        w_ = cal_w_(u, v, G_PR)
        return w_

    for (u, v, w) in G.edges(data='weight'):
        if G_PR.has_edge(u, v):
            continue
        w_ = cal_w_(u, v, G_PR)
        total_w.append(w)
        total_w_.append(w_)
    RMSE = mean_squared_error(total_w, total_w_) ** 0.5
    PCC = pearsonr(total_w, total_w_)

    return RMSE, PCC[0]




