import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="TP QDA - Master IAENG", layout="wide")
st.title("TP QDA : Frontière de Décision Quadratique")
st.subheader("EL MEHDI - Master IAENG")

# --- 1. CHARGEMENT OPTIMISÉ DU DATASET ---
@st.cache_data
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    mask = (y == 0) | (y == 1)
    return X[mask][:700], y[mask][:700]

with st.spinner("Chargement et préparation des données..."):
    X, y = load_mnist()

# --- 2. RÉDUCTION 2D (PCA) ET ENTRAÎNEMENT QDA ---
@st.cache_resource
def compute_qda_model(X_data, y_data):
    pca_2d = PCA(n_components=3)
    X_pca = pca_2d.fit_transform(X_data)
    X_2d = X_pca[:, [1, 2]] 
    
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_2d, y_data)
    
    return X_2d, qda

X_2d, qda = compute_qda_model(X, y)

# Calcul de l'accuracy globale sur l'espace 2D
y_pred = qda.predict(X_2d)
acc = accuracy_score(y, y_pred)

# --- INTERFACE CORPS ---
col_ctrl, col_visu = st.columns([1, 2])

with col_ctrl:
    st.header("⚙️ Analyse du Modèle QDA")
    st.info("💡 **Observation :** La QDA trace une frontière hyperbolique/parabolique car elle modélise une matrice de covariance propre à chaque classe.")
    
    # Affichage de l'Accuracy en gros
    st.metric(label="📊 Précision (Accuracy) du Modèle QDA", value=f"{acc * 100:.2f}%")
    
    st.markdown("---")
    st.write("Les points situés au centre montrent la zone de chevauchement entre certains '0' et '1' après réduction par ACP (PCA).")

# Grille de pixels pour la frontière
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# --- 3. RENDU ET VISUALISATION ---
with col_visu:
    st.header("📊 Frontière de Décision Quadratique (QDA)")
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    np.random.seed(42)
    idx_sample = np.random.choice(len(X_2d), 150, replace=False)
    
    scatter = ax.scatter(X_2d[idx_sample, 0], X_2d[idx_sample, 1], c=y[idx_sample], cmap='coolwarm', alpha=0.7, edgecolors='k', s=45)
    ax.set_xlabel("Composante Principale 2")
    ax.set_ylabel("Composante Principale 3")
    
    Z_qda = qda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, Z_qda, colors='blue', linewidths=3, levels=[0.5])
    
    ax.set_title(f"Frontière QDA (Accuracy : {acc * 100:.2f}%)")
    st.pyplot(fig)
    plt.close(fig)
