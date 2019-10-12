"""
    experiment for
    Leave N out
"""
import networkx as nx
import matplotlib.pyplot as plt
from utils import leave_out_n
import page_rank as PG
import fairness_goodness_computation as FG
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

per = list(range(10, 100, 10))
error = {'FG':list(range(len(per))),
         'PR':list(range(len(per))),
         'SH':list(range(len(per))),
         'TS':list(range(len(per))),
         'LR':list(range(len(per))),
         'TB':list(range(len(per))),
         'ST':list(range(len(per))),
         'RP':list(range(len(per))),
         'BD':list(range(len(per)))
         }
pcc = {'FG':list(range(len(per))),
       'PR':list(range(len(per))),
       'SH':list(range(len(per))),
       'TS':list(range(len(per))),
       'LR':list(range(len(per))),
       'TB': list(range(len(per))),
       'ST': list(range(len(per))),
       'RP': list(range(len(per))),
       'BD': list(range(len(per)))
       }

for step,n in enumerate(per):
    G_n = leave_out_n(G, n)

    PR = nx.pagerank(G_n, weight='signed_weight')
    G_PR = PG.pagerank_PR_graph(G_n, PR)
    fairness, goodness = FG.compute_fairness_goodness(G_n)
    h, a = nx.hits(G_n,max_iter=300)
    G_hits = SH.signed_G_hits(G_n, h, a)
    reg = LR.train_reg(G, G_n, fairness, goodness, G_PR, G_hits)
    bias, des =BD.compute_bias_des(G_n)
    sigma = ST.compute_status(G_n)


    error['PR'][step], pcc['PR'][step] = PG.pagerank_predict_weight(G, G_PR)
    error['FG'][step], pcc['FG'][step] = FG.FG_predict_weight(G, G_n, fairness, goodness)
    error['SH'][step], pcc['SH'][step] = SH.signed_hits(G, G_hits)
    error['TS'][step], pcc['TS'][step] = TS.triadic_status(G, G_n)
    error['BD'][step], pcc['BD'][step] = BD.BD_predict_weight(G, G_n, bias, des)
    error['RP'][step], pcc['RP'][step] = RP.reci_pred(G, G_n)
    error['ST'][step], pcc['ST'][step] = ST.status_weight_pred(G, G_n, sigma)
    error['TB'][step], pcc['TB'][step] = TB.tria_pred(G, G_n)
    error['LR'][step], pcc['LR'][step] = LR.Multiple_Regression(G, G_n, reg, fairness, goodness, G_PR, G_hits)

    print('G_len:{}, G_{}%:{}, RMSE_FG:{:.3f}, RMSE_PR:{:.3f}, RMSE_SH:{:.3f}, RMSE_TS:{:.3f}, RMSE_LR:{:.3f}, RMSE_TB:{:.3f}, RMSE_ST:{:.3f}, RMSE_RP:{:.3f}, RMSE_BD:{:.3f}'.format(
        len(G.edges()), n, len(G_n.edges()), error['FG'][step], error['PR'][step], error['SH'][step],
        error['TS'][step], error['LR'][step], error['TB'][step], error['ST'][step], error['RP'][step], error['BD'][step]))

    print('G_len:{}, G_{}%:{}, PCC_FG:{:.3f}, PCC_PR:{:.3f}, PCC_SH:{:.3f}, PCC_TS:{:.3f}, PCC_LR:{:.3f}, PCC_TB:{:.3f}, PCC_ST:{:.3f}, PCC_RP:{:.3f}, PCC_BD:{:.3f}'.format(
        len(G.edges()), n, len(G_n.edges()), pcc['FG'][step], pcc['PR'][step], pcc['SH'][step], pcc['TS'][step],
        pcc['LR'][step], pcc['TB'][step], pcc['ST'][step], pcc['RP'][step], pcc['BD'][step]))

plt.figure(dpi=500)
plt.xlim(xmax=100, xmin=0)
plt.plot(per, error['FG'], label='F&G')
plt.plot(per, error['PR'], label='PageRank')
plt.plot(per, error['SH'], label='Signed Hits')
plt.plot(per, error['TS'], label='Triadic Status')
plt.plot(per, error['LR'], label='LR')
plt.plot(per, error['TB'], label='TB')
plt.plot(per, error['ST'], label='ST')
plt.plot(per, error['RP'], label='RP')
plt.plot(per, error['BD'], label='BD')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('RMSE Error')
plt.savefig('./figures/RMSE_' + filename +'.png')

plt.figure(dpi=500)
plt.xlim(xmax=100, xmin=0)
plt.plot(per, pcc['FG'], label='F&G')
plt.plot(per, pcc['PR'], label='PageRank')
plt.plot(per, pcc['SH'], label='Signed Hits')
plt.plot(per, pcc['TS'], label='Triadic Status')
plt.plot(per, pcc['LR'], label='LR')
plt.plot(per, pcc['TB'], label='TB')
plt.plot(per, pcc['ST'], label='ST')
plt.plot(per, pcc['RP'], label='RP')
plt.plot(per, pcc['BD'], label='BD')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('PCC')
plt.savefig('./figures/PCC_' + filename +'.png')



