
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")
st.title("Dashboard Interativo – Análise de Risco de Crédito")

# Carregar modelos e dados
with open('modelo.pkl', 'rb') as f:
    modelo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)

df = pd.read_csv("df_encoded.csv")
st.success("Base de dados carregada automaticamente.")

# Transformação e predição
X = df.drop(['class', 'cluster', 'outlier'], axis=1)
X_scaled = scaler.transform(X)
y_pred = modelo.predict(X_scaled)
y_proba = modelo.predict_proba(X_scaled)[:, 1]

df['Probabilidade (good)'] = y_proba
df['Predição'] = np.where(y_pred == 1, 'good', 'bad')

# Filtros
st.sidebar.subheader("Filtros de Cliente")
risco = st.sidebar.selectbox("Classe prevista", ["Todos", "good", "bad"])
df_filtrado = df.copy()
X_filtrado = X.copy()
X_scaled_filtrado = X_scaled.copy()

if risco != "Todos":
    idxs = df[df['Predição'] == risco].index
    df_filtrado = df.loc[idxs]
    X_filtrado = X.loc[idxs]
    X_scaled_filtrado = X_scaled[idxs]

st.dataframe(df_filtrado.head(20), use_container_width=True)

# Visualização SHAP
st.subheader("Visualização SHAP para um cliente")
idx = st.selectbox("Selecione o índice do cliente", df_filtrado.index)

shap_values = explainer.shap_values(X_scaled)

if isinstance(shap_values, list):
    shap_values_instance = shap_values[1][idx]
    expected_value = explainer.expected_value[1]
else:
    shap_values_instance = shap_values[idx]
    expected_value = explainer.expected_value

st.write(f"Cliente selecionado: índice {idx}")
st.write("Waterfall plot (SHAP):")

fig, ax = plt.subplots(figsize=(10, 4))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_instance,
        base_values=expected_value,
        data=X.iloc[idx],
        feature_names=X.columns
    ),
    max_display=10, show=False
)
st.pyplot(fig)

# PCA Clusters
st.subheader("Visualização de Clusters e Outliers (PCA)")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

fig2 = plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='cluster', style='outlier', palette='Set2')
plt.title("Clusterização e Outliers")
st.pyplot(fig2)

