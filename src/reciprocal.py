from math import sqrt
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error


def reci_pred(G, G_n):

    total_w = []
    total_w_ = []
    for (u, v, w) in G.edges(data='weight'):
        if G_n.has_edge(u,v):
            continue
        if G.has_edge(v,u):
            w_ = w
        else:
            w_ = 0
        total_w.append(w)
        total_w_.append(w_)
  
    RMSE = sqrt(mean_squared_error(total_w, total_w_))
    PCC = pearsonr(total_w, total_w_)

    return RMSE, PCC[0]

