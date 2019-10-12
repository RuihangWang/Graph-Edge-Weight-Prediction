"""
    experiment for
    Leave one out
"""
import networkx as nx
from page_rank import pagerank_predict_weight, pagerank_PR_graph
import fairness_goodness_computation as FG
from signed_hits import signed_hits, signed_G_hits
from triadic_status import triadic_status
from Multiple_Regression_Model import Multiple_Regression, train_reg
import random
from sklearn.metrics import mean_squared_error


G = nx.DiGraph()

filenames = ['OTCNet', 'RFAnet', 'BTCAlphaNet', 'EpinionNetSignedNet', 'WikiSignedNet']
filename = filenames[2]

f = open('./CSV/' + filename +'.csv', "r")
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

total_w_ = {'FG':[], 'PR':[], 'SH':[], 'TS':[], 'LR':[]}
total_w = []

### remove edges
remove_edges_num = len(G.edges())
all_edges = []
for (u, v) in G.edges():
    all_edges.append((u, v))
remove_edges = random.sample(all_edges, remove_edges_num)

### G_1
G_1 = G.copy()

### cal different graphs or parameters
PR = nx.pagerank(G_1, weight='signed_weight')
G_PR = pagerank_PR_graph(G_1, PR)
fairness, goodness = FG.compute_fairness_goodness(G_1)
h, a = nx.hits(G_1)
G_hits = signed_G_hits(G_1, h, a)
reg = train_reg(G, G_1, fairness, goodness, G_PR, G_hits)

for step, (u, v) in enumerate(remove_edges):
    G_1.remove_edge(u, v)

    total_w_['PR'].append(pagerank_predict_weight(G, G_PR, (u,v)))
    total_w_['FG'].append(FG.FG_predict_weight(G, G_1, fairness, goodness, (u,v)))
    total_w_['SH'].append(signed_hits(G, G_hits, (u,v)))
    total_w_['TS'].append(triadic_status(G, G_1, (u,v)))
    total_w_['LR'].append(Multiple_Regression(G, G_1, reg, fairness, goodness, G_PR, G_hits, (u,v)))
    total_w.append(G[u][v]['weight'])

    G_1.add_edge(u,v, weight=G[u][v]['weight'],
                 signed_weight=G[u][v]['signed_weight'],
                 positive=G[u][v]['positive'],
                 negative=G[u][v]['negative'])

log = "FG:{:.3f}, PR:{:.3f}, SH:{:.3f}, TS:{:.3f}, LR:{:.3f}".format(
    mean_squared_error(total_w_['FG'], total_w) ** 0.5,
    mean_squared_error(total_w_['PR'], total_w) ** 0.5,
    mean_squared_error(total_w_['SH'], total_w) ** 0.5,
    mean_squared_error(total_w_['TS'], total_w) ** 0.5,
    mean_squared_error(total_w_['LR'], total_w) ** 0.5
)
print(log)





