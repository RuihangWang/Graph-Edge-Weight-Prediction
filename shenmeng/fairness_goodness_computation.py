import math
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error


def initialize_scores(G):
    fairness = {}
    goodness = {}
    nodes = G.nodes()
    for node in nodes:
        fairness[node] = 1
        try:
            goodness[node] = G.in_degree(node, weight='weight')*1.0/G.in_degree(node)
        except:
            goodness[node] = 0
    return fairness, goodness

def compute_fairness_goodness(G):
    fairness, goodness = initialize_scores(G)
    
    nodes = G.nodes()
    iter = 0
    while iter < 100:
        df = 0
        dg = 0
        # print('-----------------')
        # print("Iteration number", iter)
        # print('Updating goodness')
        for node in nodes:
            inedges = G.in_edges(node, data='weight')
            g = 0
            for edge in inedges:
                g += fairness[edge[0]]*edge[2]
            try:
                dg += abs(g/len(inedges) - goodness[node])
                goodness[node] = g/len(inedges)
            except:
                pass
        # print('Updating fairness')
        for node in nodes:
            outedges = G.out_edges(node, data='weight')
            f = 0
            for edge in outedges:
                f += 1.0 - abs(edge[2] - goodness[edge[1]])/2.0
            try:
                df += abs(f/len(outedges) - fairness[node])
                fairness[node] = f/len(outedges)
            except:
                pass
        
        # print('Differences in fairness score and goodness score = %.6f, %.6f' % (df, dg))
        if df < math.pow(10, -6) and dg < math.pow(10, -6):
            break
        iter+=1

    return fairness, goodness

def cal_w_(u,v,fairness,goodness):
    w_ = fairness[u] * goodness[v]
    return w_

def FG_predict_weight(G, G_n ,fairness, goodness, u_v_edge=None):

    total_w = []
    total_w_ = []

    if u_v_edge is not None:
        (u,v) = u_v_edge
        w_ = cal_w_(u, v, fairness, goodness)
        return w_

    for (u, v, w) in G.edges(data='weight'):
        if G_n.has_edge(str(u),str(v)):
            continue
        w_ = cal_w_(u, v, fairness, goodness)
        total_w.append(w)
        total_w_.append(w_)

    RMSE = mean_squared_error(total_w, total_w_) ** 0.5
    PCC = pearsonr(total_w, total_w_)

    return RMSE, PCC[0]




