import math
import csv
from src.utils import *

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

def compute_fairness_goodness(G,filename):
    fairness, goodness = initialize_scores(G)
    
    nodes = G.nodes()
    iter = 0
    while iter < 100:
        df = 0
        dg = 0
        print('-----------------')
        print("Iteration number", iter)
        print('Updating goodness')
        for node in nodes:
            inedges = G.in_edges(node, data='weight')
            g = 0
            for edge in inedges:
                g += fairness[edge[0]] * edge[2]
            try:
                dg += abs(g / len(inedges) - goodness[node])
                goodness[node] = g / len(inedges)
            except:
                pass
        print('Updating fairness')
        for node in nodes:
            outedges = G.out_edges(node, data='weight')
            f = 0
            for edge in outedges:
                f += 1.0 - abs(edge[2] - goodness[edge[1]]) / 2.0
            try:
                df += abs(f / len(outedges) - fairness[node])
                fairness[node] = f / len(outedges)
            except:
                pass

       
        with open("../results/residual_{}".format(filename),'a') as f:
            f.write(','.join([str(iter),str(df),str(dg)]) + '\n')
        
        
        print('Differences in fairness score and goodness score = %.6f, %.6f' % (df, dg))
        if df < math.pow(10, -6) and dg < math.pow(10, -6):
            break
        iter += 1
    
    return fairness, goodness

if __name__ == "__main__":

    print('\nSelect dataset for evaluation\n')
    print('Avaliable datset: BTCAlphaNet.csv, OTCNet.csv, RFAnet.csv\n')
    G = nx.DiGraph()

    path = '../dataset/'
    filename = input('Input dataset:')
    G = init_Graph(filename, path)

    # these two dictionaries have the required scores
    fairness, goodness = compute_fairness_goodness(G,filename)
    
    f = csv.writer(open('../results/F_{}'.format(filename),'w'))
    for key, val in fairness.items():
        f.writerow([key, val])

    g = csv.writer(open('../results/G_{}'.format(filename),'w'))
    for key, val in goodness.items():
        g.writerow([key, val])
    

