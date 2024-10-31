import gdown
import streamlit as st
import os
import pickle

# URL do Google Drive para sugarcane.pkl
MODEL_URL = "https://drive.google.com/uc?export=download&id=1cXHett1s4pkRKNRKKal_oguRlFkhRvqY"

@st.cache_resource
def load_model():
    model_path = "sugarcane.pkl"

    # Baixar o modelo se ele não existir no diretório
    if not os.path.exists(model_path):
        with st.spinner("Baixando o modelo..."):
            gdown.download(MODEL_URL, model_path, quiet=False)

    # Verificar se o arquivo foi baixado corretamente
    if os.path.getsize(model_path) < 1_000_000:
        st.error("Erro ao baixar o modelo. Verifique o link de download.")
        return None

    # Carregar o modelo
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    model.eval()
    return model

# Carregar o modelo
model = load_model()
