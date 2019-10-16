"""
    Experiment for
    Leave N% out
"""
import numpy as np
import pandas as pd
import sys
sys.path.append('../src')
from utils import *
from page_rank import Page_Rank
from bias_deserve import Bias_Deserve
from fairness_goodness import Fairness_Goodness
from reciprocal import Reciprocal
from signed_hits import Sighed_Hits
from status_theory import Status_Theory
from triadic_balance import Triadic_Balance
from triadic_status import Triadic_Status
from multiple_regression import Linear_Regression

print('\nSelect dataset for evaluation\n')
print('Avaliable datset: BTCAlphaNet.csv, OTCNet.csv, RFAnet.csv\n')

# input network
filename = input('Input dataset:')
G = init_Graph(filename, path='../dataset/')

if __name__ == "__main__":
    
    algorithm_type = ['PageRank', 'Bias_Deserve', 'Fairness_Goodness',
    'Reciprocal', 'Signed_HITS', 'Status_Theory',
    'Triadic_Balance', 'Triadic_Status', 'Linear_Regression']

    percentages = list(range(10, 100, 10))
    
    RMSE = {'FG':[], 'PR':[], 'SH':[], 'TS':[], 'LR':[], 'BD':[], 'RP':[], 'ST':[], 'TB':[]}
    PCC = {'FG':[], 'PR':[], 'SH':[], 'TS':[], 'LR':[], 'BD':[], 'RP':[], 'ST':[], 'TB':[]}

    for step, n in enumerate(percentages):
        G_n = leave_out_n(G, n)

        print('\nWaiting for the prediction of leaving {} out'.format(str(n)+'%'))
        PR = Page_Rank(G_n)
        BD = Bias_Deserve(G_n)
        FG = Fairness_Goodness(G_n)
        RP = Reciprocal(G_n)
        SH = Sighed_Hits(G_n)
        ST = Status_Theory(G_n)
        TB = Triadic_Balance(G_n)
        TS = Triadic_Status(G_n)
        LR = Linear_Regression(G,G_n,PR,FG,SH)

        RMSE['PR'].append(predict_weight(PR, G, G_n, (u, v))[0])
        RMSE['BG'].append(predict_weight(BG, G, G_n, (u, v))[0])
        RMSE['FG'].append(predict_weight(FG, G, G_n, (u, v))[0])
        RMSE['RP'].append(predict_weight(RP, G, G_n, (u, v))[0])
        RMSE['SH'].append(predict_weight(SH, G, G_n, (u, v))[0])
        RMSE['ST'].append(predict_weight(ST, G, G_n, (u, v))[0])
        RMSE['TB'].append(predict_weight(TB, G, G_n, (u, v))[0])
        RMSE['TS'].append(predict_weight(TS, G, G_n, (u, v))[0])
        RMSE['LR'].append(predict_weight(LR, G, G_n, (u, v))[0])

        PCC['PR'].append(predict_weight(PR, G, G_n, (u, v))[1])
        PCC['BG'].append(predict_weight(BG, G, G_n, (u, v))[1])
        PCC['FG'].append(predict_weight(FG, G, G_n, (u, v))[1])
        PCC['RP'].append(predict_weight(RP, G, G_n, (u, v))[1])
        PCC['SH'].append(predict_weight(SH, G, G_n, (u, v))[1])
        PCC['ST'].append(predict_weight(ST, G, G_n, (u, v))[1])
        PCC['TB'].append(predict_weight(TB, G, G_n, (u, v))[1])
        PCC['TS'].append(predict_weight(TS, G, G_n, (u, v))[1])
        PCC['LR'].append(predict_weight(LR, G, G_n, (u, v))[1])

    RMSE_stack = np.vstack((RMSE['PR'], RMSE['BD'], RMSE['FG'], 
                            RMSE['RP'], RMSE['SH'], RMSE['ST'], 
                            RMSE['TB'], RMSE['TS'], RMSE['LR']))

    PCC_stack = np.vstack((PCC['PR'], PCC['BD'], PCC['FG'], 
                           PCC['RP'], PCC['SH'], PCC['ST'], 
                           PCC['TB'], PCC['TS'], PCC['LR']))

    df_RMSE = pd.DataFrame(RMSE_stack, index = algorithm_type, columns = percentages)
    df_PCC = pd.DataFrame(PCC_stack, index = algorithm_type, columns = percentages)

    df_RMSE.to_csv('../results/leave_one_rmse_{}'.format(filename))
    df_PCC.to_csv('../results/leave_one_pcc_{}'.format(filename))

    print('RMSE:',df_RMSE)
    print('\nPCC:',df_PCC)










