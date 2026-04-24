# ============================================================
# APP STREAMLIT — Classification CIFAR-10
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ── Configuration de la page ────────────────────────────────
st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🔍",
    layout="centered"
)

# ── Classes CIFAR-10 ────────────────────────────────────────
CLASS_NAMES = [
    'Avion ✈️', 'Automobile 🚗', 'Oiseau 🐦', 'Chat 🐱', 'Cerf 🦌',
    'Chien 🐶', 'Grenouille 🐸', 'Cheval 🐴', 'Bateau ⛵', 'Camion 🚛'
]

# ── Chargement du modèle (mis en cache pour ne charger qu'une fois) ──
@st.cache_resource
def load_model():
    # Data augmentation (même pipeline que Colab)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")

    # Reconstruction exacte de l'architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        data_augmentation,

        # Bloc 1
        tf.keras.layers.Conv2D(32, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(32, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.25),

        # Bloc 2
        tf.keras.layers.Conv2D(64, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.30),

        # Bloc 3
        tf.keras.layers.Conv2D(128, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(128, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.40),

        # Dense
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.50),

        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Charger les poids sauvegardés
    model.load_weights('cifar10_weights.weights.h5')
    return model

# ── Prétraitement de l'image uploadée ──────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    # CIFAR-10 attend des images 32x32 RGB
    img = image.convert('RGB')
    img = img.resize((32, 32))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)
    return img_array

# ── Interface ───────────────────────────────────────────────
st.title("🔍 Classificateur d'images CIFAR-10")
st.markdown("Uploade une image et le CNN prédit sa catégorie parmi **10 classes**.")

st.divider()

# Afficher les classes disponibles
with st.expander("📋 Voir les 10 classes disponibles"):
    cols = st.columns(5)
    for i, name in enumerate(CLASS_NAMES):
        cols[i % 5].write(name)

st.divider()

# Zone d'upload
uploaded_file = st.file_uploader(
    "📁 Choisis une image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="L'image sera redimensionnée en 32x32 automatiquement"
)

if uploaded_file is not None:

    # Afficher l'image originale
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image originale")
        st.image(image, width=300)

    with col2:
        st.subheader("Image redimensionnée (32×32)")
        img_small = image.convert('RGB').resize((32, 32))
        img_display = img_small.resize((160, 160), Image.NEAREST)
        st.image(img_display, width=160)
        st.caption("C'est ce que le CNN analyse réellement")

    st.divider()

    # Prédiction
    with st.spinner("🧠 Le CNN analyse l'image..."):
        model = load_model()
        img_array = preprocess_image(image)
        # predictions = model.predict(img_array, verbose=0)[0]
        predictions = model(img_array, training=False).numpy()[0]

    # Résultat principal
    top_idx = np.argmax(predictions)
    top_conf = predictions[top_idx] * 100

    if top_conf >= 70:
        st.success(f"### ✅ Prédiction : **{CLASS_NAMES[top_idx]}**")
    elif top_conf >= 40:
        st.warning(f"### ⚠️ Prédiction : **{CLASS_NAMES[top_idx]}** (confiance modérée)")
    else:
        st.error(f"### ❌ Prédiction : **{CLASS_NAMES[top_idx]}** (confiance faible)")

    st.metric(label="Confiance", value=f"{top_conf:.1f}%")

    st.divider()

    # Top 3 prédictions avec barres de progression
    st.subheader("📊 Top 3 des prédictions")
    top3_idx = np.argsort(predictions)[::-1][:3]
    for i, idx in enumerate(top3_idx):
        conf = predictions[idx] * 100
        label = f"{'🥇' if i==0 else '🥈' if i==1 else '🥉'} {CLASS_NAMES[idx]}"
        st.write(label)
        st.progress(float(predictions[idx]))
        st.caption(f"{conf:.2f}%")

    st.divider()

    # Toutes les probabilités
    with st.expander("📈 Voir toutes les probabilités"):
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, predictions)):
            col_a, col_b = st.columns([3, 1])
            col_a.progress(float(prob), text=name)
            col_b.write(f"{prob*100:.2f}%")

# ── Footer ──────────────────────────────────────────────────
st.divider()
st.caption("CNN entraîné sur CIFAR-10 — TensorFlow/Keras — Projet Deep Learning")