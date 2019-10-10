import random

def leave_out_1(G):
    """
    leave out n percentage of edges, but will keep at least one edge of each node
    :param G: input Graph
    :param n: n percentage edges will be removed, n ranges in(0,100]
    :return: Graph after leave out n percentage edges
             the removed edge (u,v)
    """
    G_1 = G.copy()
    all_edges=[]
    for (u, v) in G.edges():
        all_edges.append((u,v))
    slice = random.sample(all_edges,len(all_edges))
    slice = slice[0]
    G_1.remove_edges_from(slice)

    return  G_1, slice[0]

def leave_out_n(G, n):
    """
    leave out n percentage of edges, but will keep at least one edge of each node
    :param G: input Graph
    :param n: n percentage edges will be removed, n ranges in(0,100]
    :return: Graph after leave out n percentage edges
    """
    G_n = G.copy()
    target = int(len(G.edges()) * (n / 100))
    all_edges=[]
    for (u, v) in G.edges():
        all_edges.append((u,v))
    slice = random.sample(all_edges,len(all_edges))
    slice = slice[:target]
    G_n.remove_edges_from(slice)

    return  G_n