import networkx as nx
import matplotlib.pyplot as plt
import fairness_goodness as FG
import bias_deserve as BD
from utils import leave_out_1, leave_out_n

G = nx.DiGraph()


f = open('WikiSignedNet.csv',"r")

for l in f:
    ls = l.strip().split(",")
    G.add_edge(ls[0], ls[1], weight = float(ls[2])) 
f.close()

error_FG = []
error_BD = []

pcc_FG = []
pcc_BD = []

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


    """
        Bias and Deserve
    """
    bias, des = BD.compute_bias_des(G_n)
    error, pcc = BD.BD_predict_weight(G, G_n, bias, des)
    error_BD.append(error)
    pcc_BD.append(pcc)

    print(error_FG[step], error_BD[step], pcc_FG[step], pcc_BD[step],)

plt.figure()
plt.xlim(xmax=100, xmin=0)
plt.plot(percentage, error_FG, color='yellow', label='F&G')
plt.plot(percentage, error_BD, color='green', label='Tidal Trust')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('RMSE Error')

#plt.savefig('./img/RMSE_' + filename +'.png')

plt.figure()
plt.xlim(xmax=100, xmin=0)
plt.plot(percentage, pcc_FG, color='yellow', label='F&G')
plt.plot(percentage, pcc_BD, color='green', label='B&D')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('PCC')

plt.show()
#plt.savefig('./img/PCC_' + filename +'.png')

