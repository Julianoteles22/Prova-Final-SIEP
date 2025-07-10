
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")

# Título
st.title("Dashboard Interativo – Análise de Risco de Crédito")

# Carregar modelo
with open('modelo.pkl', 'rb') as f:
    modelo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)

# Carregar base automaticamente
df = pd.read_csv("df_encoded.csv")
st.success("Base de dados carregada automaticamente.")

# Aplicar transformação
X = df.drop(['class', 'cluster', 'outlier'], axis=1)
X_scaled = scaler.transform(X)
y_pred = modelo.predict(X_scaled)
y_proba = modelo.predict_proba(X_scaled)[:,1]

df['Probabilidade (good)'] = y_proba
df['Predição'] = np.where(y_pred == 1, 'good', 'bad')

# Filtros
st.sidebar.subheader("Filtros de Cliente")
col1, col2 = st.sidebar.columns(2)
risco = st.sidebar.selectbox("Classe prevista", ["Todos", "good", "bad"])
if risco != "Todos":
    df = df[df['Predição'] == risco]

st.dataframe(df.head(20), use_container_width=True)

# Seleção de cliente para SHAP
st.subheader("Visualização SHAP para um cliente")
idx = st.selectbox("Selecione o índice do cliente", df.index)

shap_values = explainer.shap_values(X_scaled)

st.write(f"Cliente selecionado: índice {idx}")
st.write("Waterfall plot (SHAP):")

# Criar o objeto Explanation corretamente para o cliente selecionado
shap_exp = shap.Explanation(
    values=shap_values[:, :, 1][idx],
    base_values=explainer.expected_value[1],
    data=X.iloc[idx],
    feature_names=X.columns
)

fig, ax = plt.subplots(figsize=(10, 5))
shap.plots.waterfall(shap_exp, show=False)
st.pyplot(fig)

# Visualização: clusterização com PCA
st.subheader("Visualização de Clusters e Outliers (PCA)")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df['PCA1'] = pca_result[:,0]
df['PCA2'] = pca_result[:,1]

fig2 = plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='cluster', style='outlier', palette='Set2')
plt.title("Clusterização e Outliers")
st.pyplot(fig2)

