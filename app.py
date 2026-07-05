import streamlit as st
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="TP-QDA MNIST", page_icon="📐", layout="centered")

# --- TITRE DU TP AVEC TON NOM AU-DESSUS ET UN SAUT DE LIGNE ---
st.header("EL MEHDI - IAENG")
st.title("📐 TP-QDA : Détection du Chiffre 0")
st.write("Classification binaire avec le Dataset MNIST (0 vs Autres chiffres)")

# 1. Chargement des données (Mise en cache)
@st.cache_data
def load_data():
    # Limité à 5000 images pour que le calcul reste fluide sur Streamlit
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data[:5000] / 255.0, mnist.target[:5000]
    return X, y

X, y = load_data()

# 2. Transformation Binaire (Classe 1: '0', Classe 0: de 1 à 9)
y_binary = np.where(y == '0', 1, 0)

# Séparation des données en Entraînement / Test
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# 3. Entraînement de la QDA
@st.cache_resource
def train_qda():
    # reg_param=0.1 est indispensable sur MNIST pour stabiliser les calculs des matrices
    model = QuadraticDiscriminantAnalysis(reg_param=0.5)
    model.fit(X_train, y_train)
    return model

clf = train_qda()

# 4. Affichage des performances
st.write("### 📊 Performance du modèle")
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
st.success(f"Précision (Accuracy) globale sur l'ensemble de test : {acc * 100:.2f}%")

# 5. Interface interactive de test
st.write("### 🔍 Tester le modèle de manière interactive")
idx = st.number_input(
    "Choisissez un index d'image du dataset de test (0 à 999) :", 
    min_value=0, 
    max_value=999, 
    value=0
)

image_to_test = X_test[idx]
true_label = y_test[idx]

label_text = "C'est un 0" if true_label == 1 else "Autre chiffre"

# Affichage de l'image
st.image(
    image_to_test.reshape(28, 28), 
    caption=f"Classe réelle : {label_text}", 
    width=150
)

# Prédiction et probabilités
prediction = clf.predict([image_to_test])[0]
probabilities = clf.predict_proba([image_to_test])[0]

st.write("---")
if prediction == 1:
    st.write(f"🔮 **Prédiction du modèle :** 🟢 **C'est un 0 !** (Confiance : {probabilities[1]*100:.2f}%)")
else:
    st.write(f"🔮 **Prédiction du modèle :** 🔴 **Anomalie (Ce n'est pas un 0)** (Confiance : {probabilities[0]*100:.2f}%)")
