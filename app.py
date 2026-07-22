import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import matplotlib.patches as patches

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="TP QDA - Master IAENG", layout="wide")
st.title("TP QDA : Frontières Quadratiques & Hétérogénéité des Classes")
st.subheader("EL MEHDI - Master IAENG")

# --- 1. CHARGEMENT OPTIMISÉ DU DATASET ---
@st.cache_data
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    mask = (y == 0) | (y == 1)
    # Limitation sécurisée à 3000 images pour garantir une vitesse instantanée à l'oral
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
    st.header("⚙️ Configuration du Modèle")
    
    st.write("### 🎛️ Modes d'Analyse QDA")
    mode = st.radio("Sélectionner le test visuel :", 
                    ["Test 1 : Frontière Courbe (QDA vs LDA)", 
                     "Test 2 : Robustesse et Overfitting (Dimensions)", 
                     "Test 3 : Ellipses de Covariance (Hétérogénéité)"])
    
    st.markdown("---")
    
    if mode == "Test 1 : Frontière Courbe (QDA vs LDA)":
        st.info("💡 **Observation :** La LDA est forcée de tracer une ligne droite. La QDA, possédant une matrice de covariance par classe, génère une frontière parabolique flexible.")
    
    elif mode == "Test 2 : Robustesse et Overfitting (Dimensions)":
        dims = st.slider("Nombre de dimensions injectées dans la QDA", min_value=2, max_value=50, value=10)
        st.write(f"Dimensions actuelles : **{dims}**")
        
        # Fonction cachée pour le calcul des scores du Test 2
        @st.cache_data
        def get_dimension_scores(X_full, y_full):
            dimensions_list = [2, 5, 10, 15, 20, 30, 40]
            scores = []
            for d in dimensions_list:
                p = PCA(n_components=d)
                X_p = p.fit_transform(X_full)
                q = QuadraticDiscriminantAnalysis()
                q.fit(X_p, y_full)
                scores.append(q.score(X_p, y_full))
            return dimensions_list, scores

        dims_list, scores_list = get_dimension_scores(X, y)
        
        # Calcul direct pour la dimension choisie par le slider
        pca_var = PCA(n_components=dims)
        X_var = pca_var.fit_transform(X)
        qda_test = QuadraticDiscriminantAnalysis()
        qda_test.fit(X_var, y)
        acc = qda_test.score(X_var, y)
        
        st.success(f"Précision globale (Accuracy) : **{acc*100:.2f}%**")
        st.warning("⚠️ Attention : En haute dimension, QDA doit inverser de très grandes matrices, risquant l'overfitting.")
        
    else:
        st.info("📊 **Analyse des Variances :** Le chiffre '0' (circulaire) possède une dispersion spatiale beaucoup plus large et variée que le chiffre '1' (rectiligne). Leurs ellipses de confiance ne se ressemblent pas.")

# Grille de pixels pré-calculée pour les frontières (Test 1)
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# --- 3. RENDU ET VISUALISATION SÉCURISÉS ---
with col_visu:
    st.header("📊 Graphiques de Décision")
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Échantillon fixe pour éviter le scintillement à chaque interaction
    np.random.seed(42)
    idx_sample = np.random.choice(len(X_2d), 500, replace=False)
    ax.scatter(X_2d[idx_sample, 0], X_2d[idx_sample, 1], c=y[idx_sample], cmap='coolwarm', alpha=0.6, edgecolors='k')
    ax.set_xlabel("Composante Principale 1")
    ax.set_ylabel("Composante Principale 2")
    
    if "Test 1" in mode:
        Z_lda = lda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contour(xx, yy, Z_lda, colors='red', linewidths=2.5, levels=[0.5])
        
        Z_qda = qda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contour(xx, yy, Z_qda, colors='blue', linewidths=2.5, levels=[0.5])
        
        ax.set_title("Comparaison des frontières : LDA (Rouge, Droite) vs QDA (Bleu, Courbe)")
        st.pyplot(fig)
        plt.close(fig)
        
    elif "Test 2" in mode:
        plt.close(fig) # Fermeture de la figure principale inutile ici
        fig_dims, ax_dims = plt.subplots(figsize=(9, 4))
        ax_dims.plot(dims_list, scores_list, marker='s', color='purple', linestyle='--')
        ax_dims.set_xlabel("Nombre de dimensions (Composantes PCA)")
        ax_dims.set_ylabel("Précision du modèle")
        ax_dims.set_title("Évolution de la précision de la QDA en fonction de la dimensionnalité")
        ax_dims.grid(True, alpha=0.3)
        st.pyplot(fig_dims)
        plt.close(fig_dims)
        
    else: # Test 3 : Ellipses
        ax.set_title("Hétérogénéité spatiale des classes (Dispersion de variance)")
        
        for class_val, color, name in zip([0, 1], ['red', 'blue'], ['Chiffre 0', 'Chiffre 1']):
            X_class = X_2d[y == class_val]
            mean = np.mean(X_class, axis=0)
            cov = np.cov(X_class, rowvar=False)
            
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            
            w, h = 2 * 2 * np.sqrt(vals)
            ellipse = patches.Ellipse(xy=mean, width=w, height=h, angle=theta, 
                                      edgecolor=color, facecolor='none', linewidth=3, label=name)
            ax.add_patch(ellipse)
            
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
