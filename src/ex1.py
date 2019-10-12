"""
    experiment for
    Leave one out
"""
import networkx as nx
import random
from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr
import fairness_goodness_computation as FG
import page_rank as PG
import signed_hits as SH
import triadic_status as TS
import Multiple_Regression_Model as LR
import bias_deserve as BD
import reciprocal as RP
import status_theory as ST
import triadiac_balance as TB



G = nx.DiGraph()

filenames = ['OTCNet', 'RFAnet', 'BTCAlphaNet', 'EpinionNetSignedNet', 'WikiSignedNet']
filename = filenames[2]

f = open('./dataset/' + filename +'.csv', "r")
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

percentage = list(range(10, 100, 10))

total_w_ = {'FG':[], 'PR':[], 'SH':[], 'TS':[], 'LR':[], 'BD':[], 'RP':[], 'ST':[], 'TB':[]}
total_w = []

### remove edges
remove_edges_num = 1000
all_edges = []
for (u, v) in G.edges():
    all_edges.append((u, v))
remove_edges = random.sample(all_edges, remove_edges_num)

### G_1
G_1 = G.copy()

### cal different graphs or parameters
PR = nx.pagerank(G_1, weight='signed_weight')
G_PR = PG.pagerank_PR_graph(G_1, PR)
fairness, goodness = FG.compute_fairness_goodness(G_1)
h, a = nx.hits(G_1, max_iter=300)
G_hits = SH.signed_G_hits(G_1, h, a)
reg = LR.train_reg(G, G_1, fairness, goodness, G_PR, G_hits)
bias, des = BD.compute_bias_des(G_1)
sigma = ST.compute_status(G_1)

for step, (u, v) in enumerate(remove_edges):
    G_1.remove_edge(u, v)

    total_w_['PR'].append(PG.pagerank_predict_weight(G, G_PR, (u,v)))
    total_w_['FG'].append(FG.FG_predict_weight(G, G_1, fairness, goodness, (u,v)))
    total_w_['SH'].append(SH.signed_hits(G, G_hits, (u,v)))
    total_w_['TS'].append(TS.triadic_status(G, G_1, (u,v)))
    total_w_['BD'].append(BD.BD_predict_weight(G, G_1, bias, des, (u,v)))
    total_w_['RP'].append(RP.reci_pred(G, G_1, (u,v)))
    total_w_['ST'].append(ST.status_weight_pred(G, G_1, sigma, (u,v)))
    total_w_['TB'].append(TB.tria_pred(G, G_1, (u,v)))
    total_w_['LR'].append(LR.Multiple_Regression(G, G_1, reg, fairness, goodness, G_PR, G_hits, (u,v)))
    total_w.append(G[u][v]['weight'])

    G_1.add_edge(u,v, weight=G[u][v]['weight'],
                 signed_weight=G[u][v]['signed_weight'],
                 positive=G[u][v]['positive'],
                 negative=G[u][v]['negative'])

log_rmse = "FG:{:.3f}, PR:{:.3f}, SH:{:.3f}, TS:{:.3f}, BD:{:.3f}, RP:{:.3f}, ST:{:.3f}, TB:{:.3f}, LR:{:.3f}".format(
    mean_squared_error(total_w_['FG'], total_w) ** 0.5,
    mean_squared_error(total_w_['PR'], total_w) ** 0.5,
    mean_squared_error(total_w_['SH'], total_w) ** 0.5,
    mean_squared_error(total_w_['TS'], total_w) ** 0.5,
    mean_squared_error(total_w_['BD'], total_w) ** 0.5,
    mean_squared_error(total_w_['RP'], total_w) ** 0.5,
    mean_squared_error(total_w_['ST'], total_w) ** 0.5,
    mean_squared_error(total_w_['TB'], total_w) ** 0.5,
    mean_squared_error(total_w_['LR'], total_w) ** 0.5
)
print(log_rmse)

log_pcc = "FG:{:.3f}, PR:{:.3f}, SH:{:.3f}, TS:{:.3f}, BD:{:.3f}, RP:{:.3f}, ST:{:.3f}, TB:{:.3f}, LR:{:.3f}".format(
    pearsonr(total_w_['FG'], total_w)[0],
    pearsonr(total_w_['PR'], total_w)[0],
    pearsonr(total_w_['SH'], total_w)[0],
    pearsonr(total_w_['TS'], total_w)[0],
    pearsonr(total_w_['BD'], total_w)[0],
    pearsonr(total_w_['RP'], total_w)[0],
    pearsonr(total_w_['ST'], total_w)[0],
    pearsonr(total_w_['TB'], total_w)[0],
    pearsonr(total_w_['LR'], total_w)[0]
)
print(log_pcc)





