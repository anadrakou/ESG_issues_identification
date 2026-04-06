import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

def histogram(dataset, column):
    fig = px.histogram(dataset, x=column, nbins=30, marginal='rug', title='Distribution of '+ column)
    fig.update_layout(xaxis_title=column, yaxis_title='Frequency')
    fig.show()

def confusion_matrix(dataset):
    dataset['impact_length'] = pd.Categorical(
    dataset['impact_length'],
    categories=['Less than 2 years', '2 to 5 years', 'More than 5 years'],
    ordered=True)

    dataset['impact_level'] = pd.Categorical(
    dataset['impact_level'],
    categories=['low', 'medium', 'high'],
    ordered=True)

    conf_matrix = pd.crosstab(dataset['impact_level'], dataset['impact_length'], rownames=['Impact Level'], colnames=['Impact Length'])
    plt.figure(figsize=(10,6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix: Impact Level vs Impact Length')
    plt.show()
    return conf_matrix 