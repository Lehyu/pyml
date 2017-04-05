import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def relationship(data_df, col1, col2, plot='boxplot'):
    data = pd.concat([data_df[col1], data_df[col2]], axis=1)
    if plot == 'boxplot':
        f, ax = plt.subplots()
        fig = sns.boxplot(x=col2, y=col1, data=data)
        fig.axis(ymin=data_df[col1].min(), ymax=data_df[col2].max())
    elif plot == 'scatter':
        data.plot.scatter(x=col2, y = col1, ylim=(data_df[col1].min(), data_df[col1].max()))
    else:
        pass
    plt.show()

def hist(data, col1, col2):
    df = pd.crosstab(data[col1], data[col2])
    df.plot(kind='bar')
    plt.show()

def heatmap(data, key,k):
    corrmat = data.corr()
    cols = corrmat.nlargest(k, key)[key].index
    cm = np.corrcoef(data[cols].values.T)
    sns.set(font_scale=2.25)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                         yticklabels=cols.values,
                         xticklabels=cols.values)
    plt.xticks(rotation=90)
    plt.show()

def missing_info(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
    info = pd.concat([total, percent], axis=1, keys=['total', 'percent'])
    return info
