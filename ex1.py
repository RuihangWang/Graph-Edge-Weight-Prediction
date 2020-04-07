"""
    experiment for
    leave one out
"""
import numpy as np
import pandas as pd
from src.utils import *
from src.page_rank import Page_Rank
from src.bias_deserve import Bias_Deserve
from src.fairness_goodness import Fairness_Goodness
from src.reciprocal import Reciprocal
from src.signed_hits import Sighed_Hits
from src.status_theory import Status_Theory
from src.triadic_balance import Triadic_Balance
from src.triadic_status import Triadic_Status
from src.multiple_regression import Linear_Regression


def main():

    print('\nSelect dataset for evaluation\n')
    print('Avaliable datset: BTCAlphaNet.csv, OTCNet.csv, RFAnet.csv\n')

    filename = input('Input dataset:')
    G = init_Graph(filename, path='./dataset/')

    # remove edges
    remove_edges = leave_out_edges(G, 1000)
    # G_1

    G_1 = G.copy()

    algorithm_type = ['PageRank', 'Bias_Deserve', 'Fairness_Goodness',
                      'Reciprocal', 'Signed_HIts', 'status_Theory',
                      'Triadic_Balance', 'Triadic_status', 'Linear_Regression']

    total_w = {'fg': [], 'pr': [], 'sh': [], 'ts': [], 'lr': [], 'bd': [], 'rp': [], 'st': [], 'tb': []}
    total_w_ = {'fg': [], 'pr': [], 'sh': [], 'ts': [], 'lr': [], 'bd': [], 'rp': [], 'st': [], 'tb': []}

    rmse = {'fg': [], 'pr': [], 'sh': [], 'ts': [], 'lr': [], 'bd': [], 'rp': [], 'st': [], 'tb': []}
    pcc = {'fg': [], 'pr': [], 'sh': [], 'ts': [], 'lr': [], 'bd': [], 'rp': [], 'st': [], 'tb': []}

    print('\nWaiting for the prediction of leaving one edge out ...\n')

    pr = Page_Rank(G_1)
    bd = Bias_Deserve(G_1)
    fg = Fairness_Goodness(G_1)
    rp = Reciprocal(G_1)
    sh = Sighed_Hits(G_1)
    st = Status_Theory(G_1)
    tb = Triadic_Balance(G_1)
    ts = Triadic_Status(G_1)
    lr = Linear_Regression(G, G_1, pr, fg, sh)

    algorithm_dict = dict(pr=pr, bd=bd, fg=fg, rp=rp, sh=sh, st=st, tb=tb, ts=ts, lr=lr)

    for step, (u, v) in enumerate(remove_edges):

        G_1.remove_edge(u, v)

        for key, value in algorithm_dict.items():
            total_w[key].append(predict_weight(value, G, G_1, (u, v))[0])
            total_w_[key].append(predict_weight(value, G, G_1, (u, v))[1])

        G_1.add_edge(u, v,
                     weight=G[u][v]['weight'],
                     signed_weight=G[u][v]['signed_weight'],
                     positive=G[u][v]['positive'],
                     negative=G[u][v]['negative'])

    for key, value in algorithm_dict.items():
        rmse[key] = mean_squared_error(total_w[key], total_w_[key]) ** 0.5
        pcc[key] = pearsonr(total_w[key], total_w_[key])[0]

    rmse_stack = np.vstack(([rmse[each] for each in algorithm_dict.keys()]))
    pcc_stack = np.vstack(([pcc[each] for each in algorithm_dict.keys()]))

    df_rmse = pd.DataFrame(rmse_stack, index=algorithm_type, columns=['1'])
    df_pcc = pd.DataFrame(pcc_stack, index=algorithm_type, columns=['1'])

    df_rmse.to_csv('./results/leave_one_rmse_{}'.format(filename))
    df_pcc.to_csv('./results/leave_one_pcc_{}'.format(filename))

    print('rmse:', df_rmse)
    print('\npcc:', df_pcc)


if __name__ == "__main__":
    main()
