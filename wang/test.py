import networkx as nx
import matplotlib.pyplot as plt
import fairness_goodness as FG
import bias_deserve as BD
from utils import leave_out_1, leave_out_n

G = nx.DiGraph()

f = open('example.csv',"r")

for l in f:
    ls = l.strip().split(",")
    G.add_edge(ls[0], ls[1], weight = float(ls[2])) 
f.close()

bias, des = BD.compute_bias_des(G)

'''
The test bias and deserve should be:
        Node 1   Node 2  Node 3
Bias  |   0.13 |  0.08 |  -0.14
Des   |  -0.33 |  0.73 |  -1

'''
print(bias,des)