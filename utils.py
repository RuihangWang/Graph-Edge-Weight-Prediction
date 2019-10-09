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
    target = len(G.edges()) * (1 - n / 100)
    len_G_n_edges = len(G_n.edges())

    all_edges=[]
    for (u, v) in G.edges():
        all_edges.append((u,v))
    slice = random.sample(all_edges,len(all_edges))

    while (running):
        for (u, v) in slice:
            if G_n.has_edge(u, v) and (len(G_n.in_edges(u)) >=1 or len(G_n.out_edges(u)) >= 2) and (len(G_n.in_edges(v)) >=2 or len(G_n.out_edges(v)) >= 1):
                G_n.remove_edge(u,v)
                len_G_n_edges -= 1
                if len_G_n_edges <= target:
                    running=False
                    break
    return  G_n