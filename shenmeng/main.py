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
from tidal_trust import tidal_trust

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
error_tidal_trust = []

pcc_FG = []
pcc_PR = []
pcc_signed_hits = []
pcc_tidal_trust = []
for step,n in enumerate(percentage):
    G_n = leave_out_n(G, n)

    """
        PageRank
    """
    PR = nx.pagerank(G_n, weight='signed_weight')
    error, pcc = pagerank_predict_weight(G, G_n, PR)
    error_PR.append(error)
    pcc_PR.append(pcc)

    """
        Fairness and Goodness
    """
    fairness, goodness = FG.compute_fairness_goodness(G_n)
    error, pcc = FG.FG_predict_weight(G, G_n, fairness, goodness)
    error_FG.append(error)
    pcc_FG.append(pcc)

    """
        Signed hits
    """
    h, a = nx.hits(G)
    error, pcc = signed_hits(G, G_n, h, a)
    error_signed_hits.append(error)
    pcc_signed_hits.append(pcc)

    """
        Tidal Trust
    """
    error, pcc = tidal_trust(G, G_n)
    error_tidal_trust.append(error)
    pcc_tidal_trust.append(pcc)

    print('G_len:{}, G_{}%:{}, RMSE_FG:{:.3f}, RMSE_PR:{:.3f}, RMSE_SH:{:.3f}, RMSE_TR:{:.3f} '.format(
        len(G.edges()), n, len(G_n.edges()), error_FG[step], error_PR[step], error_signed_hits[step], error_tidal_trust[step]))

    print('G_len:{}, G_{}%:{}, PCC_FG:{:.3f}, PCC_PR:{:.3f}, PCC_SH:{:.3f}, PCC_TR:{:.3f} '.format(
        len(G.edges()), n, len(G_n.edges()), pcc_FG[step], pcc_PR[step], pcc_signed_hits[step], pcc_tidal_trust[step]))

plt.figure(dpi=500)
plt.xlim(xmax=100, xmin=0)
plt.plot(percentage, error_FG, color='blue', label='F&G')
plt.plot(percentage, error_PR, color='red', label='PageRank')
plt.plot(percentage, error_signed_hits, color='green', label='Signed Hits')
plt.plot(percentage, error_tidal_trust, color='black', label='Tidal Trust')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('RMSE Error')
plt.savefig('./img/RMSE_' + filename +'.png')

plt.figure(dpi=500)
plt.xlim(xmax=100, xmin=0)
plt.plot(percentage, pcc_FG, color='blue', label='F&G')
plt.plot(percentage, pcc_PR, color='red', label='PageRank')
plt.plot(percentage, pcc_signed_hits, color='green', label='Signed Hits')
plt.plot(percentage, pcc_tidal_trust, color='black', label='Tidal Trust')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('PCC')
plt.savefig('./img/PCC_' + filename +'.png')



