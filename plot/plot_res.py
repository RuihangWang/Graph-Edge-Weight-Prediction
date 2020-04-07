import pandas as pd
import plotly.graph_objects as go

print("\nPossible results: residual_+'dataset' \n")
print("e.g. residual_OTCNet.csv")
print('\nAvaliable datset: BTCAlphaNet.csv, OTCNet.csv, RFAnet.csv\n')

path = '../results/'
filename = input('Input results:')

try:
    df = pd.read_csv(path + filename, header = None)

    F_res = list(df[1])[1:]
    G_res = list(df[2])[1:]

    iter = df[0]

    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=F_res, y=iter, name='Fairness',
                            line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=G_res, y=iter, name='Goodness',
                            line=dict(color='palevioletred', width=4)))
    # dash options include 'dash', 'dot', and 'dashdot'

    # Edit the layout
    fig.update_layout(title='Fairness and Goodness Res. of {}'.format(filename[9:-4]),
                        xaxis_title='Iteration Number',
                        yaxis_title='Change in Value')
    fig.write_image('../results/Res_{}.png'.format(filename[9:-4]))

    fig.show()

except:
    print('None such a result')