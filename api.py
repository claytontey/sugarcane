import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import requests
import pickle
import os

# URL do modelo no Google Drive (substitua "SEU_ID_DO_ARQUIVO" pelo ID real)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1cXHett1s4pkRKNRKKal_oguRlFkhRvqY/view?usp=sharing"


@st.cache_resource
def load_model():
    model_path = "sugarcane.pkl"
    
    # Baixar o modelo se não estiver no diretório atual
    if not os.path.exists(model_path):
        with st.spinner("Baixando o modelo..."):
            response = requests.get(MODEL_URL)
            with open(model_path, "wb") as f:
                f.write(response.content)

    # Verificar o tamanho do arquivo
    if os.path.getsize(model_path) < 1_000_000:  # Exemplo: tamanho mínimo de 1 MB
        st.error("Erro ao baixar o modelo. Verifique o link de download e tente novamente.")
        return None

    # Carregar o modelo
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    model.eval()
    return model

# Carregar o modelo
model = load_model()
if model is None:
    st.stop()  # Interrompe a execução se o modelo não for carregado corretamente

# Definir classes de doenças
classes = ['Healthy', 'Mosaic', 'RedHot', 'Rust', 'Yellow']

# Função de pré-processamento da imagem
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Interface do Streamlit
st.title("Classificação de Doenças na Folha da Cana-de-Açúcar")
st.write("Faça o upload de uma imagem de uma folha de cana para classificar a doença.")

# Upload de imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagem Carregada', use_column_width=True)
    
    # Pré-processar a imagem
    image_tensor = preprocess_image(image)
    
    # Realizar a previsão, se o modelo foi carregado corretamente
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()
        predicted_class = classes[prediction]
    
    # Exibir o resultado
    st.write(f"**Classe Predita:** {predicted_class}")
