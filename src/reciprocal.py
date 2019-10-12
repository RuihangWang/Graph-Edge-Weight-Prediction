from math import sqrt
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error

def cal_w_(G_n,u,v):
    if G_n.has_edge(v,u):
        w = G_n[v][u]['weight']
        w_ = w
    else:
        w_ = 0
    return w_

def reci_pred(G, G_n, u_v_edge=None):

    if u_v_edge is not None:
        u,v = u_v_edge
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

