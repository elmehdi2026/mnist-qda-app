import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="TP QDA - Master IAENG", layout="wide")
st.title("TP QDA : Frontières Quadratiques & Hétérogénéité des Classes")
st.subheader("EL MEHDI - Master IAENG")

# --- 1. CHARGEMENT OPTIMISÉ DU DATASET ---
@st.cache_data
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    # Sélection des chiffres 0 et 1 (formes géométriques très différentes)
    mask = (y == 0) | (y == 1)
    return X[mask], y[mask]

X, y = load_mnist()

# --- 2. RÉDUCTION 2D POUR LA VISUALISATION DE LA FRONTIÈRE ---
# On réduit à 2 dimensions via PCA pour pouvoir tracer un plan X, Y
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X)

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
        
        # Entraînement à dimensions variables
        pca_var = PCA(n_components=dims)
        X_var = pca_var.fit_transform(X)
        
        qda_test = QuadraticDiscriminantAnalysis()
        qda_test.fit(X_var, y)
        acc = qda_test.score(X_var, y)
        st.success(f"Précision globale (Accuracy) : **{acc*100:.2f}%**")
        st.warning("⚠️ Attention : En haute dimension, QDA doit inverser de très grandes matrices, risquant l'overfitting.")
        
    else:
        st.info("📊 **Analyse des Variances :** Le chiffre '0' (circulaire) possède une dispersion spatiale beaucoup plus large et variée que le chiffre '1' (rectiligne). Leurs ellipses de confiance ne se ressemblent pas.")

# --- 3. LOGIQUE MATHÉMATIQUE ET TRACÉ DES FRONTIÈRES (Pour Test 1 et 3) ---
# Entraînement des deux modèles sur l'espace 2D
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

lda.fit(X_2d, y)
qda.fit(X_2d, y)

# Création d'une grille de pixels pour dessiner les zones de décision
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# --- 4. RENDU ET VISUALISATION ---
with col_visu:
    st.header("📊 Graphiques de Décision")
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Affichage des points de données (Échantillon de 500 points pour la lisibilité)
    idx_sample = np.random.choice(len(X_2d), 500, replace=False)
    scatter = ax.scatter(X_2d[idx_sample, 0], X_2d[idx_sample, 1], c=y[idx_sample], cmap='coolwarm', alpha=0.6, edgecolors='k')
    ax.set_xlabel("Composante Principale 1")
    ax.set_ylabel("Composante Principale 2")
    
    if "Test 1" in mode:
        # Frontière LDA
        Z_lda = lda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contour(xx, yy, Z_lda, colors='red', linewidths=2.5, levels=[0.5])
        
        # Frontière QDA
        Z_qda = qda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contour(xx, yy, Z_qda, colors='blue', linewidths=2.5, levels=[0.5])
        
        ax.set_title("Comparaison des frontières : LDA (Rouge, Droite) vs QDA (Bleu, Courbe)")
        st.pyplot(fig)
        
    elif "Test 2" in mode:
        # Graphique simple des scores
        fig_dims, ax_dims = plt.subplots(figsize=(9, 4))
        dimensions_list = [2, 5, 10, 15, 20, 30, 40]
        scores = []
        for d in dimensions_list:
            p = PCA(n_components=d)
            X_p = p.fit_transform(X)
            q = QuadraticDiscriminantAnalysis()
            q.fit(X_p, y)
            scores.append(q.score(X_p, y))
            
        ax_dims.plot(dimensions_list, scores, marker='s', color='purple', linestyle='--')
        ax_dims.set_xlabel("Nombre de dimensions (Composantes PCA)")
        ax_dims.set_ylabel("Précision du modèle")
        ax_dims.set_title("Évolution de la précision de la QDA en fonction de la dimensionnalité")
        ax_dims.grid(True, alpha=0.3)
        st.pyplot(fig_dims)
        
    else: # Test 3 : Ellipses
        import matplotlib.patches as patches
        ax.set_title("Hétérogénéité spatiale des classes (Dispersion de variance)")
        
        # Tracer l'enveloppe simplifiée pour chaque classe (0 et 1)
        for class_val, color, name in zip([0, 1], ['red', 'blue'], ['Chiffre 0', 'Chiffre 1']):
            X_class = X_2d[y == class_val]
            mean = np.mean(X_class, axis=0)
            cov = np.cov(X_class, rowvar=False)
            
            # Extraction des valeurs propres pour dessiner l'ellipse de dispersion
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            
            # Dessin de l'ellipse de confiance (2 écarts-types)
            w, h = 2 * 2 * np.sqrt(vals)
            ellipse = patches.Ellipse(xy=mean, width=w, height=h, angle=theta, 
                                      edgecolor=color, facecolor='none', linewidth=3, label=name)
            ax.add_patch(ellipse)
            
        ax.legend()
        st.pyplot(fig)
