from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error

def cal_w_(u,v,G_n):
    iter_triad = 0
    w_ = 0
    for x in G_n.succ[u]:
        if G_n.has_edge(x, v):
            iter_triad += 1
            w_ += G_n[u][x]['weight'] + G_n[x][v]['weight']
        if G_n.has_edge(v, x):
            iter_triad += 1
            w_ += G_n[u][x]['weight'] - G_n[v][x]['weight']
    for x in G_n.pred[u]:
        if G_n.has_edge(x, v):
            iter_triad += 1
            w_ += G_n[x][v]['weight'] - G_n[x][u]['weight']
        if G_n.has_edge(v, x):
            iter_triad += 1
            w_ += -(G_n[v][x]['weight'] + G_n[x][u]['weight'])
    if iter_triad != 0:
        w_ /= iter_triad
    return w_

def triadic_status(G,G_n, u_v_edge=None):

    total_w = []
    total_w_ = []

    if u_v_edge is not None:
        (u, v) = u_v_edge
        return cal_w_(u,v,G_n)

    for (u,v,w) in G.edges(data='weight'):
        if G_n.has_edge(u,v):
            continue
        w_ = cal_w_(u, v, G_n)

        total_w.append(w)
        total_w_.append(w_)

    RMSE = mean_squared_error(total_w, total_w_) ** 0.5
    PCC = pearsonr(total_w, total_w_)

    return RMSE, PCC[0]


