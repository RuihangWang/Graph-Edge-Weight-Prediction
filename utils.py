import random

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