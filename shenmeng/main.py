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
from page_rank import pagerank_predict_weight
import fairness_goodness_computation as FG
from signed_hits import signed_hits

G = nx.DiGraph()

filenames = ['OTCNet', 'RFAnet', 'BTCAlphaNet', 'EpinionNetSignedNet', 'WikiSignedNet']
filename = filenames[2]

f = open('./CSV/' + filename +'.csv', "r")
for l in f:
    ls = l.strip().split(",")
    if float(ls[2]) >= 0:
        w = 1
    else:
        w = 0
    G.add_edge(ls[0], ls[1], weight=float(ls[2]), signed_weight=w)

f.close()

percentage = list(range(10, 100, 10))
error_FG = []
error_PR = []
error_signed_hits = []

for step,n in enumerate(percentage):
    G_n = leave_out_n(G, n)
    print('G_n')
    """
        PageRank
    """
    PR = nx.pagerank(G_n, weight='signed_weight')
    error = pagerank_predict_weight(G, G_n, PR)
    error_PR.append(error)

    """
        Fairness and Goodness
    """
    fairness, goodness = FG.compute_fairness_goodness(G_n)
    error = FG.FG_predict_weight(G, G_n, fairness, goodness)
    error_FG.append(error)

    """
        Signed hits
    """
    h, a = nx.hits(G)
    error = signed_hits(G, G_n, h, a)
    error_signed_hits.append(error)

    print('G_len:{}, G_{}%:{}, RMSE_FG:{:.3f}, RMSE_PR:{:.3f}, RMSE_SH:{:.3f} '.format(
        len(G.edges()), n, len(G_n.edges()), error_FG[step], error_PR[step], error_signed_hits[step]))

plt.figure(dpi=500)
plt.plot(percentage, error_FG, color='blue', label='F&G')
plt.plot(percentage, error_PR, color='red', label='PageRank')
plt.plot(percentage, error_signed_hits, color='green', label='Signed Hits')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('RMSE Error')
plt.savefig('./img/result_' + filename +'.png')



