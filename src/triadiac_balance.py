from math import sqrt
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error

def cal_w_(G_n, u,v):
    neigh_u = set(G_n.pred[u])|set(G_n.succ[u])
    neigh_v = set(G_n.pred[v])|set(G_n.succ[v])
    neigh = neigh_u & neigh_v
    w_u = []
    w_v = []
    if len(neigh) != 0:
        for nu in list(neigh):
            if G_n.has_edge(u, nu):
                w_u.append(G_n[u][nu]['weight'])
            if G_n.has_edge(nu, u):
                w_u.append(G_n[nu][u]['weight'])
        for nv in list(neigh):
            if G_n.has_edge(v, nv):
                w_v.append(G_n[v][nv]['weight'])
            if G_n.has_edge(nv, v):
                w_v.append(G_n[nv][v]['weight'])

        num = len(w_u + w_v)
        total = sum(w_u + w_v)
        w_ = total / num
    else:
        w_ = 0
    return w_

def tria_pred(G, G_n, u_v_edge=None):

    if u_v_edge is not None:
        u, v = u_v_edge
        return cal_w_(G_n, u, v)

    total_w = []
    total_w_ = []
    for (u, v, w) in G.edges(data='weight'):
        if G_n.has_edge(u,v):
            continue
        w_ = cal_w_(G_n, u,v)
        total_w.append(w)
        total_w_.append(w_)
  
    RMSE = sqrt(mean_squared_error(total_w, total_w_))
    PCC = pearsonr(total_w, total_w_)

    return RMSE, PCC[0]
