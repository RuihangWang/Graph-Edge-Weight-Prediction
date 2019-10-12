from math import sqrt
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error



def compute_status(G): 
    sigma = {}
    for node in G.nodes():
        inedges = G.in_edges(node, data='weight')
        outedges = G.out_edges(node, data='weight')

        w_p_in = 0.0
        w_n_in = 0.0
        w_p_out = 0.0
        w_n_out = 0.0

        p_in = 0
        n_in = 0
        p_out = 0
        n_out = 0
        
        for edge in inedges:
            if edge[2]>0:
                w_p_in += edge[2]
                p_in += 1
            elif edge[2]<0:
                w_n_in += edge[2]
                n_in += 1
        
        for edge in outedges:
            if edge[2]>0:
                w_p_out += edge[2]
                p_out += 1
            else:
                w_n_out += edge[2]
                n_out += 1
        try:
            sigma[node] = abs(w_p_in/p_in) - abs(w_n_in/n_in) + abs(w_n_out/n_out) - abs(w_p_out/p_out)
        except:
            sigma[node] = 0
    
    return sigma

def status_weight_pred(G, G_n, sigma):
    
    total_w = []
    total_w_ = []

    for u, v, w in G.edges(data = 'weight'):
        if G_n.has_edge(u, v):
            continue
        w_ = sigma[u] - sigma[v]
        total_w.append(w)
        total_w_.append(w_)
  
    RMSE = sqrt(mean_squared_error(total_w, total_w_))
    PCC = pearsonr(total_w, total_w_)

    return RMSE, PCC[0]

