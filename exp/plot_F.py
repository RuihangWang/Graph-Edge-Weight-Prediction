import pandas as pd
import numpy as np
import plotly.graph_objects as go 

print("\nPossible results: F_+'dataset' \n")
print("e.g. F_BTCAlphaNet.csv")
print('\nAvaliable datset: BTCAlphaNet.csv, OTCNet.csv, RFAnet.csv\n')

path = '../results/'
filename = input('Input results:')

try:
    df = pd.read_csv(path + filename, header = None)

    F = list(df[1])

    x = range(0,20)
    x = [each/20 for each in x]
    frac = []
    for i,each_range in enumerate(x):
        min_ = each_range
        max_ = each_range + 2/len(x)
        frac.append(0)
        for each in F:
            if each < max_ and each>min_:
                frac[-1] += 1
    
    frac = [each/sum(frac) for each in frac] 

    np.savetxt("../results/F_distribution_{}".format(filename[2:]), np.vstack((x,frac)),
                delimiter=',')

    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=x, y=frac,
                            line=dict(color='rosybrown', width=4)))
    # dash options include 'dash', 'dot', and 'dashdot'

    # Edit the layout
    fig.update_layout(title='Fairness distribution of {}'.format(filename[2:-4]),
                        xaxis_title='Fairness score',
                        yaxis_title='Frac of vertices with Fairness f')
    fig.write_image('../results/F_{}.png'.format(filename[2:-4]))

    fig.show()

except:
    print('None such a result')