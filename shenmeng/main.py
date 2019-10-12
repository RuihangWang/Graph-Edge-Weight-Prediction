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
from Multiple_Regression_Model import Multiple_Regression, train_reg
import time

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

per = list(range(10, 100, 10))
error = {'FG':list(range(len(per))), 'PR':list(range(len(per))), 'SH':list(range(len(per))), 'TS':list(range(len(per))), 'LR':list(range(len(per)))}
pcc = {'FG':list(range(len(per))), 'PR':list(range(len(per))), 'SH':list(range(len(per))), 'TS':list(range(len(per))), 'LR':list(range(len(per)))}

for step,n in enumerate(per):
    G_n = leave_out_n(G, n)

    PR = nx.pagerank(G_n, weight='signed_weight')
    G_PR = pagerank_PR_graph(G_n, PR)
    fairness, goodness = FG.compute_fairness_goodness(G_n)
    h, a = nx.hits(G_n,max_iter=300)
    G_hits = signed_G_hits(G_n, h, a)
    reg = train_reg(G, G_n, fairness, goodness, G_PR, G_hits)

    error['PR'][step], pcc['PR'][step] = pagerank_predict_weight(G, G_PR)
    error['FG'][step], pcc['FG'][step] = FG.FG_predict_weight(G, G_n, fairness, goodness)
    error['SH'][step], pcc['SH'][step] = signed_hits(G, G_hits)
    error['TS'][step], pcc['TS'][step] = triadic_status(G, G_n)
    error['LR'][step], pcc['LR'][step] = Multiple_Regression(G, G_n, reg, fairness, goodness, G_PR, G_hits)

    print('G_len:{}, G_{}%:{}, RMSE_FG:{:.3f}, RMSE_PR:{:.3f}, RMSE_SH:{:.3f}, RMSE_TS:{:.3f}, RMSE_LR:{:.3f}'.format(
        len(G.edges()), n, len(G_n.edges()), error['FG'][step], error['PR'][step], error['SH'][step], error['TS'][step], error['LR'][step]))

    print('G_len:{}, G_{}%:{}, PCC_FG:{:.3f}, PCC_PR:{:.3f}, PCC_SH:{:.3f}, PCC_TS:{:.3f}, PCC_LR:{:.3f}'.format(
        len(G.edges()), n, len(G_n.edges()), pcc['FG'][step], pcc['PR'][step], pcc['SH'][step], pcc['TS'][step], pcc['LR'][step]))

plt.figure(dpi=500)
plt.xlim(xmax=100, xmin=0)
plt.plot(per, error['FG'], label='F&G')
plt.plot(per, error['PR'], label='PageRank')
plt.plot(per, error['SH'], label='Signed Hits')
plt.plot(per, error['TS'], label='Triadic Status')
plt.plot(per, error['LR'], label='LR')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('RMSE Error')
plt.savefig('./img/RMSE_' + filename +'.png')

plt.figure(dpi=500)
plt.xlim(xmax=100, xmin=0)
plt.plot(per, pcc['FG'], label='F&G')
plt.plot(per, pcc['PR'], label='PageRank')
plt.plot(per, pcc['SH'], label='Signed Hits')
plt.plot(per, pcc['TS'], label='Triadic Status')
plt.plot(per, pcc['LR'], label='LR')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('PCC')
plt.savefig('./img/PCC_' + filename +'.png')



