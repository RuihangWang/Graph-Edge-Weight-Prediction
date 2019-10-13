class Status_Theory():
    def __init__(self,G):
        self.G = G
        sigma = {}
        for node in G.nodes():
            inedges = G.in_edges(node, data='weight')
            outedges = G.out_edges(node, data='weight')

            w_p_in = 0.0
            w_n_in = 0.0
            w_p_out = 0.0
            w_n_out = 0.0

            p_in = 0
            n_in = 0
            p_out = 0
            n_out = 0

            for edge in inedges:
                if edge[2] > 0:
                    w_p_in += edge[2]
                    p_in += 1
                elif edge[2] < 0:
                    w_n_in += edge[2]
                    n_in += 1

            for edge in outedges:
                if edge[2] > 0:
                    w_p_out += edge[2]
                    p_out += 1
                else:
                    w_n_out += edge[2]
                    n_out += 1
            try:
                sigma[node] = abs(w_p_in / p_in) - abs(w_n_in / n_in) + abs(w_n_out / n_out) - abs(w_p_out / p_out)
            except:
                sigma[node] = 0

        self.sigma = sigma

    def cal_w_(self, u, v):
        return self.sigma[u] - self.sigma[v]
