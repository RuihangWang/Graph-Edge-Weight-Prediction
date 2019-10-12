'''
Code for the paper:
Edge Weight Prediction in Weighted Signed Networks.
Conference: ICDM 2016
Authors: Srijan Kumar, Francesca Spezzano, VS Subrahmanian and Christos Faloutsos

Author of code: Srijan Kumar
Email of code author: srijan@cs.stanford.edu
'''
import networkx as nx
import matplotlib.pyplot as plt
from utils import leave_out_n
from page_rank import pagerank_predict_weight, pagerank_PR_graph
import fairness_goodness_computation as FG
from signed_hits import signed_hits, signed_G_hits
from triadic_status import triadic_status
from Multiple_Regression_Model import Multiple_Regression
import random

G = nx.DiGraph()

filenames = ['OTCNet', 'RFAnet', 'BTCAlphaNet', 'EpinionNetSignedNet', 'WikiSignedNet']
filename = filenames[0]

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

error_FG = []
error_PR = []
error_signed_hits = []
error_triadic_status = []
error_LR = []

pcc_FG = []
pcc_PR = []
pcc_signed_hits = []
pcc_triadic_status = []
pcc_LR = []

remove_edges_num = 1000
remove_edges = range(remove_edges_num)
all_edges = []
for (u, v) in G.edges():
    all_edges.append((u, v))
remove_edges = random.sample(all_edges,len(remove_edges))

for step, (u, v) in enumerate(remove_edges):
    G_1 = G.copy()
    G_1.remove_edge(u,v)

    """
        PageRank
    """
    PR = nx.pagerank(G_1, weight='signed_weight')
    G_PR = pagerank_PR_graph(G_1, PR)
    w_PR = pagerank_predict_weight(G, G_PR, (u,v))

    """
        Fairness and Goodness
    """
    fairness, goodness = FG.compute_fairness_goodness(G_1)
    w_FG = FG.FG_predict_weight(G, G_1, fairness, goodness, (u,v))

    """
        Signed hits
    """
    h, a = nx.hits(G)
    G_hits = signed_G_hits(G_1, h, a)
    w_SH = signed_hits(G, G_hits, (u,v))

    """
        Triadic Status
    """
    w_TS = triadic_status(G, G_1, (u,v))

    """
        Multiple_Regression
    """
    w_LR = Multiple_Regression(G, G_1, fairness, goodness, G_PR, G_hits, (u,v))



plt.figure(dpi=500)
plt.xlim(xmax=100, xmin=0)
plt.plot(percentage, error_FG, label='F&G')
plt.plot(percentage, error_PR, label='PageRank')
plt.plot(percentage, error_signed_hits, label='Signed Hits')
plt.plot(percentage, error_triadic_status, label='Triadic Status')
plt.plot(percentage, error_LR, label='LR')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('RMSE Error')
plt.savefig('./img/RMSE_' + filename +'.png')

plt.figure(dpi=500)
plt.xlim(xmax=100, xmin=0)
plt.plot(percentage, pcc_FG, label='F&G')
plt.plot(percentage, pcc_PR, label='PageRank')
plt.plot(percentage, pcc_signed_hits, label='Signed Hits')
plt.plot(percentage, pcc_triadic_status, label='Triadic Status')
plt.plot(percentage, pcc_LR, label='LR')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('PCC')
plt.savefig('./img/PCC_' + filename +'.png')



