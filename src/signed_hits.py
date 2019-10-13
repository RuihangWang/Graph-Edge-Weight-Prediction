import networkx as nx
from collections import Counter

def signed_hits(G, max_iter=100, tol=1.0e-8, normalized=True):

    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise Exception("hits() not defined for graphs with multiedges.")
    if len(G) == 0:
        return {}, {}
    # choose fixed starting vector if not given

    h_p = dict.fromkeys(G, 1.0 / G.number_of_nodes())
    h_n = dict.fromkeys(G, - 1.0 / G.number_of_nodes())
    h = h_p
    a = h_p
    for _ in range(max_iter):  # power iteration: make up to max_iter iterations
        h_p_last = h_p
        h_n_last = h_n

        h_p = dict.fromkeys(h_p_last.keys(), 0)
        h_n = dict.fromkeys(h_n_last.keys(), 0)
        a_p = dict.fromkeys(h_p_last.keys(), 0)
        a_n = dict.fromkeys(h_n_last.keys(), 0)

        # this "matrix multiply" looks odd because it is
        # doing a left multiply a^T=hlast^T*G

        for u in h_p:
            for v in G.pred[u]:
                if G[v][u]['weight'] >= 0 :
                    a_p[u] += h_p_last[v] * G[v][u]['weight']
                else :
                    a_n[u] -= h_n_last[v] * G[v][u]['weight']
        for u in h_p:
            for v in G.succ[u]:
                if G[u][v]['weight'] >= 0:
                    h_p[u] += a_p[v] * G[u][v]['weight']
                else :
                    h_n[u] -= a_n[v] * G[u][v]['weight']


        # normalize vector
        s = 1.0 / max(h_p.values())
        for n in h_p:
            h_p[n] *= s
        # normalize vector
        s = -1.0 / min(h_n.values())
        for n in h_n:
            h_n[n] *= s
        # normalize vector
        s = 1.0 / max(a_p.values())
        for n in a_p:
            a_p[n] *= s
        # normalize vector
        s = -1.0 / min(a_n.values())
        for n in a_n:
            a_n[n] *= s

        for key, value in h.items():
            h[key] = h_p[key] - h_n[key]
            a[key] = a_p[key] - a_n[key]

        # check convergence, l1 norm
        err = sum([abs(h_p[n] - h_p_last[n]) for n in h_p] + [abs(h_n[n] - h_n_last[n]) for n in h_n] )
        if err < tol:
            break
    else:
        raise nx.PowerIterationFailedConvergence(max_iter)
    if normalized:
        s = 1.0 / sum(a.values())
        for n in a:
            a[n] *= s
        s = 1.0 / sum(h.values())
        for n in h:
            h[n] *= s
    return h, a

class Sighed_Hits():
    def __init__(self, G):
        self.G = G
        h, a = signed_hits(G, max_iter=500)
        G_hits = self.G.copy()
        w_in = {}
        w_out = {}
        Hits_total_out = {}
        Hits_total_in = {}

        for (u, w) in G_hits.nodes(data='weight'):
            w_in_v = 0
            w_out_v = 0
            Hits_total_out_v = 0
            Hits_total_in_v = 0
            for edge in G_hits.out_edges(u, data='weight'):
                w_out_v += edge[2] * h[edge[0]]
                Hits_total_out_v += h[edge[0]]
            for edge in G_hits.in_edges(u, data='weight'):
                w_in_v += edge[2] * a[edge[1]]
                Hits_total_in_v += a[edge[1]]
            w_in[u] = w_in_v
            w_out[u] = w_out_v
            Hits_total_in[u] = Hits_total_in_v
            Hits_total_out[u] = Hits_total_out_v
        nx.set_node_attributes(G_hits, w_in, 'w_in')
        nx.set_node_attributes(G_hits, w_out, 'w_out')
        nx.set_node_attributes(G_hits, Hits_total_in, 'Hits_total_in')
        nx.set_node_attributes(G_hits, Hits_total_out, 'Hits_total_out')

        self.G_hits = G_hits

    def cal_w_(self, u, v):

        w_ = 0
        if self.G_hits.node[u]['Hits_total_out'] != 0:
            w_ += self.G_hits.node[u]['w_out'] / self.G_hits.node[u]['Hits_total_out']
        if self.G_hits.node[v]['Hits_total_in'] != 0:
            w_ += self.G_hits.node[v]['w_in'] / self.G_hits.node[v]['Hits_total_in']
        w_ /= 2
        return w_