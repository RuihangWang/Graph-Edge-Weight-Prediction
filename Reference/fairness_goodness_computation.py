'''
Code for the paper:
Edge Weight Prediction in Weighted Signed Networks. 
Conference: ICDM 2016
Authors: Srijan Kumar, Francesca Spezzano, VS Subrahmanian and Christos Faloutsos

Author of code: Srijan Kumar
Email of code author: srijan@cs.stanford.edu
'''
import networkx as nx
import math
import random
import matplotlib.pyplot as plt

def initialize_scores(G):
    fairness = {}
    goodness = {}
    nodes = G.nodes()
    for node in nodes:
        fairness[node] = 1
        try:
            goodness[node] = G.in_degree(node, weight='weight')*1.0/G.in_degree(node)
        except:
            goodness[node] = 0
    return fairness, goodness

def compute_fairness_goodness(G):
    fairness, goodness = initialize_scores(G)
    
    nodes = G.nodes()
    iter = 0
    while iter < 100:
        df = 0
        dg = 0
        # print('-----------------')
        # print("Iteration number", iter)
        # print('Updating goodness')
        for node in nodes:
            inedges = G.in_edges(node, data='weight')
            g = 0
            for edge in inedges:
                # print(fairness[edge[0]])
                # print(edge)
                # print(edge[2])
                g += fairness[edge[0]]*edge[2]
            try:
                dg += abs(g/len(inedges) - goodness[node])
                goodness[node] = g/len(inedges)
            except:
                pass
        # print('Updating fairness')
        for node in nodes:
            outedges = G.out_edges(node, data='weight')
            f = 0
            for edge in outedges:
                f += 1.0 - abs(edge[2] - goodness[edge[1]])/2.0
            try:
                df += abs(f/len(outedges) - fairness[node])
                fairness[node] = f/len(outedges)
            except:
                pass
        
        # print('Differences in fairness score and goodness score = %.6f, %.6f' % (df, dg))
        if df < math.pow(10, -6) and dg < math.pow(10, -6):
            break
        iter+=1

    return fairness, goodness

# skip = int(sys.argv[1])
def compute_edge_weight_FG(G, fairness, goodness):

    RMSE = 0
    iter = 0
    for (u, v, true_weight) in G.edges(data='weight'):
        predict_weight = fairness[u] * goodness[v]
        RMSE += (predict_weight - true_weight) ** 2
        iter += 1
    RMSE /= iter
    RMSE = RMSE ** 0.5

    return RMSE

def leave_out_n(G, n):
    """
    leave out n percentage of edges, but will keep at least one edge of each node
    :param G: input Graph
    :param n: n percentage edges will be removed, n ranges in(0,100]
    :return: Graph after leave out n percentage edges
    """

    G_n = G.copy()
    running = True
    len_G_edges = len(G.edges())
    len_G_n_edges = len(G_n.edges())
    while (running):
        for (u, v) in G.edges():
            if (len(G_n.in_edges(u)) + len(G_n.out_edges(u))) > 1 and (len(G_n.in_edges(v)) + len(G_n.out_edges(v))):
                if G_n.has_edge(u, v) and random.random() <= n / 100:
                    G_n.remove_edge(u,v)
                    len_G_n_edges -= 1
                if (len_G_edges * (1 - n / 100)) >= len_G_n_edges:
                    running=False
                    break



    return  G_n

"""
Fairness and Goodness
"""
G = nx.DiGraph()
# source target rating time
f = open("soc-sign-bitcoinalpha.csv","r")
for l in f:
    ls = l.strip().split(",")
    G.add_edge(ls[0], ls[1], weight = float(ls[2])/10) ## the weight should already be in the range of -1 to 1

f.close()

percentage = list(range(0, 91, 3))
errors = []

for n in percentage:

    G_n = leave_out_n(G, n)
    fairness, goodness = compute_fairness_goodness(G_n)
    error = compute_edge_weight_FG(G, fairness, goodness)
    errors.append(error)
    print('G_len:{}, G_{}%:{}, RMSE_error:{}'.format(len(G.edges()), n, len(G_n.edges()), error))

"""
Page Rank
"""
G_pr = nx.DiGraph()

f = open("soc-sign-bitcoinalpha.csv","r")
for l in f:
    ls = l.strip().split(",")
    w = (int(ls[2]) - (-10)) / 20
    G_pr.add_edge(ls[0], ls[1], weight = w) ## the weight should already be in the range of -1 to 1

f.close()

page_rank = nx.pagerank(G_pr, alpha=0.85, weight='weight')


plt.figure(dpi=500)
plt.plot(percentage,errors,color='blue',label='F&G')
plt.legend()
plt.xlabel('Percentage of edges removed')
plt.ylabel('RMSE Error')
plt.show()
plt.savefig('result.png')



