import networkx as nx
from scipy.stats.stats import pearsonr


def tidal_trust(G, G_n):

    RMSE = 0
    iter = 0
    total_w = []
    total_w_ = []
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
        except:
            pass

        RMSE += (w_ - w) ** 2
        iter += 1
        total_w.append(w)
        total_w_.append(w_)

    RMSE /= iter
    RMSE = RMSE ** 0.5
    PCC = pearsonr(total_w, total_w_)


    return RMSE, PCC[0]








