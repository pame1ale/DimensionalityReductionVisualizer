from flask import Flask, render_template, jsonify
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import json

app = Flask(__name__)

# Cargar el conjunto de datos wine de scikit-learn
data = load_wine()
wine_df = pd.DataFrame(data.data, columns=data.feature_names)
wine_df['target'] = data.target


@app.route('/')
def index():
   return render_template('index.html')

@app.route('/pca_plot')
def pca_plot():
    # Aplicar PCA para reducir la dimensionalidad a 2 componentes principales
    pca = PCA(n_components=2)
    wine_sintarget = wine_df.drop(columns=['target'])
    reduced_data = pca.fit_transform(wine_sintarget)

    # Crear un DataFrame con los datos reducidos
    reduced_df = pd.DataFrame(data=reduced_data, columns=['Component 1', 'Component 2'])
    reduced_df['target'] = wine_df['target']

    # Crear un gráfico de dispersión interactivo con Plotly
    fig = go.Figure()
    for target_label in reduced_df['target'].unique():
        fig.add_trace(go.Scatter(
            x=reduced_df.loc[reduced_df['target'] == target_label, 'Component 1'],
            y=reduced_df.loc[reduced_df['target'] == target_label, 'Component 2'],
            mode='markers',
            name=f'Cluster {target_label}'
        ))

    fig.update_layout(
        #title='PCA - Wine Data',
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        showlegend=True
    )

    # Convertir los datos del gráfico a formato JSON compatible

    plot_data = json.loads(fig.to_json())

    return jsonify(plot_data)

@app.route('/tsne_plot')
def tsne_plot():
    # Aplicar t-SNE para reducir la dimensionalidad a 2 componentes
    tsne = TSNE(n_components=2, random_state=42)
    wine_sintarget = wine_df.drop(columns=['target'])
    reduced_data = tsne.fit_transform(wine_sintarget)

    # Crear un DataFrame con los datos reducidos
    reduced_df = pd.DataFrame(data=reduced_data, columns=['Component 1', 'Component 2'])
    reduced_df['target'] = wine_df['target']

    # Crear un gráfico de dispersión interactivo con Plotly
    fig = go.Figure()
    for target_label in reduced_df['target'].unique():
        fig.add_trace(go.Scatter(
            x=reduced_df.loc[reduced_df['target'] == target_label, 'Component 1'],
            y=reduced_df.loc[reduced_df['target'] == target_label, 'Component 2'],
            mode='markers',
            name=f'Cluster {target_label}'
        ))

    fig.update_layout(
        #title='t-SNE - Wine Data',
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        showlegend=True
    )

    # Convertir los datos del gráfico a formato JSON compatible
    plot_data = json.loads(fig.to_json())

    return jsonify(plot_data)


if __name__ == '__main__':
   app.run(debug=True)