# ============================================================
# APP STREAMLIT — Classification CIFAR-10 (EfficientNetB0)
# ============================================================

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🔍",
    layout="centered"
)

CLASS_NAMES = [
    'Avion ✈️', 'Automobile 🚗', 'Oiseau 🐦', 'Chat 🐱', 'Cerf 🦌',
    'Chien 🐶', 'Grenouille 🐸', 'Cheval 🐴', 'Bateau ⛵', 'Camion 🚛'
]

IMG_SIZE = 96

# ── Reconstruction du modèle + chargement des poids ────────
@st.cache_resource
def load_model():
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,              # pas besoin d'imagenet ici
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = True

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.load_weights('efficientnet_cifar10.weights.h5')
    return model

# ── Prétraitement de l'image uploadée ──────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype('float32')   # [0, 255] — preprocess_input s'en occupe
    return np.expand_dims(img_array, axis=0)       # (1, 96, 96, 3)

# ── Interface ───────────────────────────────────────────────
st.title("🔍 Classificateur d'images CIFAR-10")
st.markdown("Modèle **EfficientNetB0** pré-entraîné sur ImageNet, fine-tuné sur CIFAR-10 — **89% de précision**")

with st.expander("📋 Les 10 classes détectables"):
    cols = st.columns(5)
    for i, name in enumerate(CLASS_NAMES):
        cols[i % 5].write(name)

st.divider()

uploaded_file = st.file_uploader(
    "📁 Choisis une image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image originale")
        st.image(image, width=300)
    with col2:
        st.subheader(f"Redimensionnée ({IMG_SIZE}×{IMG_SIZE})")
        img_small = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        img_display = img_small.resize((192, 192), Image.NEAREST)
        st.image(img_display, width=192)
        st.caption("Ce que le modèle analyse")

    st.divider()

    with st.spinner("🧠 Analyse en cours..."):
        model = load_model()
        img_array = preprocess_image(image)
        predictions = model(img_array, training=False).numpy()[0]

    top_idx  = int(np.argmax(predictions))
    top_conf = predictions[top_idx] * 100

    if top_conf >= 70:
        st.success(f"### ✅ {CLASS_NAMES[top_idx]}")
    elif top_conf >= 40:
        st.warning(f"### ⚠️ {CLASS_NAMES[top_idx]} (confiance modérée)")
    else:
        st.error(f"### ❌ {CLASS_NAMES[top_idx]} (confiance faible)")

    st.metric("Confiance", f"{top_conf:.1f}%")
    st.divider()

    st.subheader("📊 Top 3")
    top3 = np.argsort(predictions)[::-1][:3]
    for i, idx in enumerate(top3):
        medal = ['🥇','🥈','🥉'][i]
        st.write(f"{medal} {CLASS_NAMES[idx]}")
        st.progress(float(predictions[idx]))
        st.caption(f"{predictions[idx]*100:.2f}%")

    with st.expander("📈 Toutes les probabilités"):
        for name, prob in zip(CLASS_NAMES, predictions):
            col_a, col_b = st.columns([3, 1])
            col_a.progress(float(prob), text=name)
            col_b.write(f"{prob*100:.2f}%")

st.divider()
st.caption("EfficientNetB0 • Transfer Learning • CIFAR-10 • TensorFlow/Keras")