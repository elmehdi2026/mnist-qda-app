import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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
    # Réduction du nombre d'images pour alléger le nuage de points (divisé par deux)
    return X[mask][:1500], y[mask][:1500]

with st.spinner("Chargement et préparation des données..."):
    X, y = load_mnist()

# --- 2. RÉDUCTION 2D (PCA) ET ENTRAÎNEMENT QDA ---
@st.cache_resource
def compute_qda_model(X_data, y_data):
    # Utilisation des composantes 1 et 3 pour accentuer la courbure et la non-linéarité
    pca_2d = PCA(n_components=3)
    X_pca = pca_2d.fit_transform(X_data)
    X_2d = X_pca[:, [0, 2]] 
    
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_2d, y_data)
    
    return X_2d, qda

X_2d, qda = compute_qda_model(X, y)

# --- INTERFACE CORPS ---
col_ctrl, col_visu = st.columns([1, 2])

with col_ctrl:
    st.header("⚙️ Analyse du Modèle QDA")
    st.info("💡 **Observation :** Contrairement à la LDA, la QDA calcule une matrice de covariance spécifique pour chaque classe. Cela lui permet de dessiner une **frontière parabolique (courbe)** capable d'épouser la forme des données[cite: 12].")
    st.markdown("---")
    st.write("Le nuage de points a été allégé pour une visibilité parfaite et sans surcharge visuelle pendant l'oral.")

# Grille de pixels pour tracer la frontière courbe
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

# --- 3. RENDU ET VISUALISATION ---
with col_visu:
    st.header("📊 Frontière de Décision Quadratique (QDA)")
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Échantillon allégé pour un affichage propre (divisé par deux par rapport à avant)
    np.random.seed(42)
    idx_sample = np.random.choice(len(X_2d), 300, replace=False)
    
    # Affichage du nuage de points
    scatter = ax.scatter(X_2d[idx_sample, 0], X_2d[idx_sample, 1], c=y[idx_sample], cmap='coolwarm', alpha=0.7, edgecolors='k', s=40)
    ax.set_xlabel("Composante Principale 1")
    ax.set_ylabel("Composante Principale 3")
    
    # Tracé de la frontière QDA sous forme de courbe parabolique
    Z_qda = qda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, Z_qda, colors='blue', linewidths=3, levels=[0.5])
    
    ax.set_title("Frontière de Décision QDA (Courbe Parabolique)")
    st.pyplot(fig)
    plt.close(fig)
