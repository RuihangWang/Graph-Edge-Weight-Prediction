import networkx as nx
import matplotlib.pyplot as plt
from utils import leave_out_1, leave_out_n
from math import sqrt
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from utils import leave_out_1, leave_out_n

def tria_pred(G, G_n):

    total_w = []
    total_w_ = []
    for (u, v, w) in G.edges(data='weight'):
        if G_n.has_edge(u,v):
            continue
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

            # for nu in list(neigh):
            #     try:
            #         w_u.append(G[nu][u]['weight'])
            #     except:
            #         w_u.append(G[u][nu]['weight'])
                
            # for nv in list(neigh):
            #     try:
            #         w_v.append(G[nv][v]['weight'])
            #     except:
            #         w_v.append(G[v][nv]['weight'])
            num = len(w_u + w_v)
            total = sum(w_u + w_v)
            w_ = total / num
        else:
            w_ = 0
    
        total_w.append(w)
        total_w_.append(w_)
  
    RMSE = sqrt(mean_squared_error(total_w, total_w_))
    PCC = pearsonr(total_w, total_w_)

    return RMSE, PCC[0]

G = nx.DiGraph()

f = open('RFAnet.csv',"r")

for l in f:
    ls = l.strip().split(",")
    G.add_edge(ls[0], ls[1], weight = float(ls[2])) 
f.close()

total_w = []
total_w_ = []

error_Tria = []

pcc_Tria = []

percentage = list(range(10, 100, 10))

for step, n in enumerate(percentage):

    G_n = leave_out_n(G, n)

    """
        Triadic Balance
    """
    error, pcc = tria_pred(G, G_n)
    error_Tria.append(error)
    pcc_Tria.append(pcc)
    
    print(error_Tria[step], pcc_Tria[step])

plt.figure()
plt.xlim(xmax=100, xmin=0)
#plt.plot(percentage, error_FG, color='yellow', label='F&G')
plt.plot(percentage, error_Tria, color='blue', label='Triadic Balance')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('RMSE Error')

plt.figure()
plt.xlim(xmax=100, xmin=0)
#plt.plot(percentage, pcc_FG, color='yellow', label='F&G')
plt.plot(percentage, pcc_Tria, color='blue', label='Triadic Balance')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('PCC')

plt.show()