"""
Page Rank
"""
import networkx as nx

class Page_Rank:
    def __init__(self, G):

        PR = nx.pagerank(G, weight='signed_weight')
        G_PR = G.copy()

        w_in = {}
        w_out = {}
        PR_total_out = {}
        PR_total_in = {}

        # 对G_PR图的所有节点n计算out edge和in edge的边权重在PageRank值上的加权平均
        for (u, w) in G_PR.nodes(data='weight'):
            w_in_v = 0
            w_out_v = 0
            PR_total_out_v = 0
            PR_total_in_v = 0
            for edge in G.out_edges(u, data='weight'):
                w_out_v += edge[2] * PR[edge[1]]
                PR_total_out_v += PR[edge[1]]
            for edge in G.in_edges(u, data='weight'):
                w_in_v += edge[2] * PR[edge[1]]
                PR_total_in_v += PR[edge[1]]
            w_in[u] = w_in_v
            w_out[u] = w_out_v
            PR_total_in[u] = PR_total_in_v
            PR_total_out[u] = PR_total_out_v
        nx.set_node_attributes(G_PR, w_in, 'w_in')
        nx.set_node_attributes(G_PR, w_out, 'w_out')
        nx.set_node_attributes(G_PR, PR_total_in, 'PR_total_in')
        nx.set_node_attributes(G_PR, PR_total_out, 'PR_total_out')

        self.G_PR = G_PR

    def cal_w_(self, u, v):
        w_ = self.G_PR.node[u]['w_out'] + self.G_PR.node[v]['w_in']
        PR_total = self.G_PR.node[u]['PR_total_out'] + self.G_PR.node[v]['PR_total_in']
        if PR_total != 0:
            w_ /= PR_total
        return w_



