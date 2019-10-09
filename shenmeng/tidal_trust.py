import networkx as nx
import matplotlib.pyplot as plt
from itertools import compress
import numpy as np
from utils import leave_out_n


def tidal_trust(G, G_n):

    RMSE = 0
    iter = 0
    for (u, v, w) in G.edges(data='weight'):
        if G_n.has_edge(u, v):
            continue

        paths = nx.all_shortest_paths(G_n, u, v)
        weight_positive = 0
        weight_negative = 0
        w_ = 0
        iter_weight = 0

        try:
            for p in paths:

                path_length = len(p)
                weight_of_the_path_p = []
                weight_of_the_path_n = []

                for step, node in enumerate(p):
                    if step + 1 >= path_length:
                        break
                    node_next = p[step+1]
                    weight_p = G.get_edge_data(node, node_next)['positive']
                    weight_n = G.get_edge_data(node, node_next)['negative']

                    weight_of_the_path_p.append(weight_p)
                    weight_of_the_path_n.append(weight_n)

                for i in range(len(weight_of_the_path_p)):
                    weight_positive = weight_of_the_path_p[i]
                    if i + 1 >= len(weight_of_the_path_p):
                        break
                    if weight_of_the_path_p[i + 1]>weight_of_the_path_p[i]:
                        weight_positive = None
                        break

                for i in range(len(weight_of_the_path_n)):
                    weight_negative = weight_of_the_path_n[i]
                    if i + 1 >= len(weight_of_the_path_n):
                        break
                    if weight_of_the_path_n[i + 1] < weight_of_the_path_n[i]:
                        weight_negative = None
                        break

                if weight_positive is not None and weight_negative is not None:
                    w_ += weight_positive + weight_negative
                    iter_weight += 1
            if iter_weight != 0:
                w_ /= iter_weight
            else:
                w_ = 0
            RMSE += (w_ - w) ** 2
            iter += 1
        except:
            pass
        print(w, w_)
    RMSE /= iter
    RMSE = RMSE ** 0.5

    return RMSE

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

    G.add_edge(int(ls[0]), int(ls[1]), weight=float(ls[2]), signed_weight=w, positive=p, negative=n)

f.close()

G_n = leave_out_n(G, 20)
error = tidal_trust(G, G_n)
print(error)
# nx.set_node_attributes(G,0,name='max')
#
# node_max=[]
# q = []
# color = [0] * len(G.edges())

# tidal_trust(G, 1, 591)







