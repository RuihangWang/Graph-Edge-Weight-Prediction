import networkx as nx
import matplotlib.pyplot as plt
import fairness_goodness as FG
from math import sqrt
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from utils import leave_out_1, leave_out_n

def PageRank_Pred(G, G_n, PR):
    for (u, v, w) in G.edges(data='weight'):
        if G_n.has_edge(u,v):
            continue
        w_ = PR[u] - PR[v]
        total_w_.append(w_)
        total_w.append(w)

    RMSE = sqrt(mean_squared_error(total_w, total_w_))
    PCC = pearsonr(total_w, total_w_)

    return RMSE, PCC[0]


G = nx.DiGraph()

f = open('RFAnet.csv',"r")

for l in f:
    ls = l.strip().split(",")
    G.add_edge(ls[0], ls[1], weight = ((float(ls[2])+1.0)/2.0) )
f.close()


total_w = []
total_w_ = []

error_FG = []
error_PR = []

pcc_FG = []
pcc_PR = []

percentage = list(range(10, 100, 10))

for step, n in enumerate(percentage):

    G_n = leave_out_n(G, n)
    """
        Fairness and Goodness
    """
    fairness, goodness = FG.compute_fairness_goodness(G_n)
    error, pcc = FG.FG_predict_weight(G, G_n, fairness, goodness)
    error_FG.append(error)
    pcc_FG.append(pcc)

    PR = nx.pagerank(G_n, alpha = 0.85, weight='weight')
    error, pcc = PageRank_Pred(G, G_n, PR)
    error_PR.append(error)
    pcc_PR.append(pcc)
    
    print(error_FG[step], pcc_FG[step], error_PR[step], pcc_PR[step])

plt.figure()
plt.xlim(xmax=100, xmin=0)
plt.plot(percentage, error_FG, color='yellow', label='F&G')
plt.plot(percentage, error_PR, color='blue', label='PR')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('RMSE Error')

plt.figure()
plt.xlim(xmax=100, xmin=0)
plt.plot(percentage, pcc_FG, color='yellow', label='F&G')
plt.plot(percentage, pcc_PR, color='blue', label='PR')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('PCC')

plt.show()




    