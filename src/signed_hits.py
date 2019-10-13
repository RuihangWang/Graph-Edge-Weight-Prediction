import networkx as nx

class Sighed_Hits():
    def __init__(self, G):
        self.G = G
        h, a = nx.hits(G, max_iter=500)
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