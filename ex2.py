"""
    experiment for
    leave N% out
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

    # input network
    filename = input('Input dataset:')
    G = init_Graph(filename, path='../dataset/')

    algorithm_type = ['PageRank', 'Bias_Deserve', 'Fairness_Goodness',
                      'Reciprocal', 'Signed_HITS', 'Status_Theory',
                      'Triadic_Balance', 'Triadic_Status', 'Linear_Regression']

    algorithm_list = ['fg', 'pr', 'sh', 'ts', 'lr', 'bd', 'rp', 'st', 'tb']

    percentages = list(range(10, 100, 10))

    rmse = {'fg': [], 'pr': [], 'sh': [], 'ts': [], 'lr': [], 'bd': [], 'rp': [], 'st': [], 'tb': []}
    pcc = {'fg': [], 'pr': [], 'sh': [], 'ts': [], 'lr': [], 'bd': [], 'rp': [], 'st': [], 'tb': []}

    for step, n in enumerate(percentages):

        G_n = leave_out_n(G, n)

        print('\nWaiting for the prediction of leaving {} out'.format(str(n) + '%'))

        pr = Page_Rank(G_n)
        bd = Bias_Deserve(G_n)
        fg = Fairness_Goodness(G_n)
        rp = Reciprocal(G_n)
        sh = Sighed_Hits(G_n)
        st = Status_Theory(G_n)
        tb = Triadic_Balance(G_n)
        ts = Triadic_Status(G_n)
        lr = Linear_Regression(G, G_n, pr, fg, sh)

        algorithm_dict = dict(pr=pr, bd=bd, fg=fg, rp=rp, sh=sh, st=st, tb=tb, ts=ts, lr=lr)

        for key, value in algorithm_dict.items():
            rmse[key].append(predict_weight(value, G, G_n)[0])
            pcc[key].append(predict_weight(value, G, G_n)[1])

    rmse_stack = np.vstack(([rmse[each] for each in algorithm_list]))
    pcc_stack = np.vstack(([pcc[each] for each in algorithm_list]))

    df_rmse = pd.DataFrame(rmse_stack, index=algorithm_type, columns=percentages)
    df_pcc = pd.DataFrame(pcc_stack, index=algorithm_type, columns=percentages)

    df_rmse.to_csv('../results/leave_N_rmse_{}'.format(filename))
    df_pcc.to_csv('../results/leave_N_pcc_{}'.format(filename))

    print('rmse:', df_rmse)
    print('\npcc:', df_pcc)


if __name__ == "__main__":
    main()
