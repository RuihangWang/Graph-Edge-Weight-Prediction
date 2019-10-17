import networkx as nx
from src.bias_deserve import Bias_Deserve

G = nx.DiGraph()

f = open('test.csv',"r")
for l in f:
    ls = l.strip().split(",")
    G.add_edge(ls[0], ls[1], weight = float(ls[2]))
f.close()

BD = Bias_Deserve(G)
bias, des = BD.compute_bias_des(G)


# The test bias and deserve should be:
#         Node 1   Node 2  Node 3
# Bias  |   0.13 |  0.08 |  -0.14
# Des   |  -0.33 |  0.73 |  -1

'''
A. Mishra and A. Bhattacharya, “Finding the bias and prestige of nodes
in networks based on trust scores,” in WWW, 2011.
'''

print(bias, des)

