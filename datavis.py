import numpy as np
import time
import pandas as pd
### code mostly from towardsdatascience.com

# For plotting
import plotly.io as plt_io
import plotly.graph_objects as go

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import utils
import keras_uncertainty

def save_image (fig, name):
    import os

    if not os.path.exists("images"):
        os.mkdir("images")

    fig.write_image(f"images/PCA_{name}.png")


def plot_2d(component1, component2, Y, name):

    translate_name = {
        "simple_mlp": "MLP",
        "convnet" : "ConvNet",
        "mc_dropconnect_convnet" : "MC-DropConnect ConvNet",
        "mc_dropout_convnet" : "MC-Dropout ConvNet",
        "mc_dropconnect_simple_mlp" : "MC-DropConnect MLP",
        "mc_dropout_simple_mlp" : "MC-Dropout MLP",
        "ens_convnet_extractor" : "ConvNet Deep-Ensemble",
        "ens_simple_mlp_extractor" : "MLP Deep-Ensemble"
    }
    title_name = translate_name [name]
    fig = go.Figure(data=go.Scatter(
        x = component1,
        y = component2,
        mode='markers',
        marker=dict(
            size=20,
            color=Y, #set color equal to a variable
            colorscale='Rainbow', # one of plotly colorscales
            showscale=True,
            line_width=1
        ),
    ))
    fig.update_layout(margin=dict( l=100,r=100,b=100,t=100),width=2000,height=1200,
                      title=f'PCA plot of features obtained by {title_name}, with class labels',
                      font=dict(
                          family="Courier New, monospace",
                          size=30,
                          color="black")
                      )
    fig.layout.template = 'ggplot2'

    fig.show()
    save_image (fig, name)


def pca_plot(x, Y, name):
    from sklearn.preprocessing import StandardScaler
    ## Standardizing the data
    x = StandardScaler().fit_transform(x)
    start = time.time()
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    print('Duration: {} seconds'.format(time.time() - start))
    principal = pd.DataFrame(data = principalComponents
                             , columns = ['principal component 1', 'principal component 2','principal component 3'])
    plot_2d(principalComponents[:, 0],principalComponents[:, 1], Y, name)


def visualize_data (extractor, fwd_passes=None, samples = 1000, name= "model"):
    X, Y, tests, domain, ishape, nclasses = utils.load_fashion_data (samples)
    if extractor.__class__ is keras_uncertainty.models.DeepEnsembleRegressor:
        features = extractor.predict (X) [0]
    elif fwd_passes is None :
        features = extractor.predict (X)
    else:
        features = extractor.predict (X, fwd_passes) [0]

    Y = np.argmax (Y, axis=1)
    np.array (map (str, Y))

    pca_plot(features, Y, name)
