import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="TP QDA - Master IAENG", layout="wide")
st.title("TP QDA : Frontières Quadratiques (LDA vs QDA)")
st.subheader("EL MEHDI - Master IAENG")

# --- 1. CHARGEMENT OPTIMISÉ DU DATASET ---
@st.cache_data
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    mask = (y == 0) | (y == 1)
    # Limitation sécurisée à 3000 images pour une fluidité totale à l'oral
    return X[mask][:3000], y[mask][:3000]

with st.spinner("Chargement et préparation des données..."):
    X, y = load_mnist()

# --- 2. RÉDUCTION 2D ET ENTRAÎNEMENT MIS EN CACHE ---
@st.cache_resource
def compute_models(X_data, y_data):
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_data)
    
    lda = LinearDiscriminantAnalysis()
    qda = QuadraticDiscriminantAnalysis()
    lda.fit(X_2d, y_data)
    qda.fit(X_2d, y_data)
    
    return X_2d, lda, qda

X_2d, lda, qda = compute_models(X, y)

# --- INTERFACE CORPS ---
col_ctrl, col_visu = st.columns([1, 2])

with col_ctrl:
    st.header("⚙️ Analyse du Modèle")
    st.info("💡 **Observation :** La LDA est forcée de tracer une ligne droite. La QDA, possédant une matrice de covariance par classe, génère une frontière parabolique flexible[cite: 12].")
    st.markdown("---")
    st.write("Ce graphique met en évidence la capacité de la QDA à s'adapter à la variance réelle des données spatiales des chiffres MNIST (0 et 1)[cite: 12].")

# Grille de pixels pré-calculée pour les frontières
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# --- 3. RENDU ET VISUALISATION SÉCURISÉS ---
with col_visu:
    st.header("📊 Frontières de Décision 2D")
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Échantillon fixe pour l'affichage
    np.random.seed(42)
    idx_sample = np.random.choice(len(X_2d), 500, replace=False)
    ax.scatter(X_2d[idx_sample, 0], X_2d[idx_sample, 1], c=y[idx_sample], cmap='coolwarm', alpha=0.6, edgecolors='k')
    ax.set_xlabel("Composante Principale 1")
    ax.set_ylabel("Composante Principale 2")
    
    # Tracé des frontières LDA (Droite) et QDA (Courbe)
    Z_lda = lda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, Z_lda, colors='red', linewidths=2.5, levels=[0.5])
    
    Z_qda = qda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(xx, yy, Z_qda, colors='blue', linewidths=2.5, levels=[0.5])
    
    ax.set_title("Comparaison des frontières : LDA (Rouge, Droite) vs QDA (Bleu, Courbe)")
    st.pyplot(fig)
    plt.close(fig)
