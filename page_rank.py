"""
Page Rank
"""

def pagerank_predict_weight(G, G_n, PR):
    RMSE = 0
    iter = 0
    for (u, v, w) in G.edges(data='signed_weight'):
        if G_n.has_edge(str(u), str(v)):
            continue

        w_ = 0
        PR_total = 0
        for edge in G_n.out_edges(str(u),data='signed_weight'):
            w_ += edge[2] * PR[str(edge[1])]
            PR_total += PR[str(edge[1])]
        for edge in G_n.in_edges(str(v),data='signed_weight'):
            w_ += edge[2] * PR[str(edge[0])]
            PR_total += PR[str(edge[0])]
        if PR_total != 0 :
            w_ /= PR_total

        iter += 1
        RMSE += (w_ - w) ** 2
    RMSE /= iter
    RMSE = RMSE ** 0.5
    return RMSE




