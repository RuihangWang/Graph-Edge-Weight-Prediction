import networkx as nx
import matplotlib.pyplot as plt
from utils import leave_out_1, leave_out_n
from math import sqrt
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from utils import leave_out_1, leave_out_n

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

G = nx.DiGraph()


f = open('RFAnet.csv',"r")

for l in f:
    ls = l.strip().split(",")
    G.add_edge(ls[0], ls[1], weight = float(ls[2])) 
f.close()

total_w = []
total_w_ = []

error_Reci = []

pcc_Reci = []


percentage = list(range(10, 100, 10))

for step, n in enumerate(percentage):

    G_n = leave_out_n(G, n)

    """
        Reciprocal
    """
    error, pcc = reci_pred(G, G_n)
    error_Reci.append(error)
    pcc_Reci.append(pcc)
    
    print(error_Reci[step], pcc_Reci[step])

plt.figure()
plt.xlim(xmax=100, xmin=0)
#plt.plot(percentage, error_FG, color='yellow', label='F&G')
plt.plot(percentage, error_Reci, color='blue', label='PR')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('RMSE Error')

plt.figure()
plt.xlim(xmax=100, xmin=0)
#plt.plot(percentage, pcc_FG, color='yellow', label='F&G')
plt.plot(percentage, pcc_Reci, color='blue', label='PR')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('PCC')

plt.show()
