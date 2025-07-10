
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

# Carregando os arquivos
with open('modelo.pkl', 'rb') as f:
    modelo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)

df = pd.read_csv("df_encoded.csv")
st.success("Base de dados carregada automaticamente.")

# Previsão inicial
X_full = df.drop(['class', 'cluster', 'outlier'], axis=1)
X_scaled_full = scaler.transform(X_full)
y_pred_full = modelo.predict(X_scaled_full)
y_proba_full = modelo.predict_proba(X_scaled_full)[:, 1]

df['Probabilidade (good)'] = y_proba_full
df['Predição'] = np.where(y_pred_full == 1, 'good', 'bad')

# Filtros interativos
st.sidebar.subheader("Filtros de Cliente")
risco = st.sidebar.selectbox("Classe prevista", ["Todos", "good", "bad"])

df_filtered = df.copy()
if risco != "Todos":
    df_filtered = df[df['Predição'] == risco]

# Atualiza X e X_scaled com base no filtro
X = df_filtered.drop(['class', 'cluster', 'outlier', 'Probabilidade (good)', 'Predição'], axis=1)
X_scaled = scaler.transform(X)

st.dataframe(df_filtered.head(20), use_container_width=True)

# Visualização SHAP
st.subheader("Visualização SHAP para um cliente")
idx = st.selectbox("Selecione o índice do cliente", df_filtered.index)

shap_values = explainer.shap_values(X_scaled)
st.write(f"Cliente selecionado: índice {idx}")
st.write("Waterfall plot (SHAP):")

shap_exp = shap.Explanation(values=shap_values[:, :, 1][df_filtered.index.get_loc(idx)],
                            base_values=explainer.expected_value[1],
                            data=X.loc[idx])

fig = shap.plots._waterfall.waterfall_legacy(shap_exp, show=False)
st.pyplot(fig)

# Visualização PCA
st.subheader("Visualização de Clusters e Outliers (PCA)")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df_filtered['PCA1'] = pca_result[:, 0]
df_filtered['PCA2'] = pca_result[:, 1]

fig2 = plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_filtered, x='PCA1', y='PCA2', hue='cluster', style='outlier', palette='Set2')
plt.title("Clusterização e Outliers")
st.pyplot(fig2)

