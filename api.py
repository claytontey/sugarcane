import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import pickle
import numpy as np

# Configurar o dispositivo (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Título e descrição
st.title("Classificação de Doenças na Folha da Cana-de-Açúcar")
st.write("Faça o upload de uma imagem de uma folha de cana para classificar a doença.")

# Carregar o modelo treinado e mover para o dispositivo
@st.cache_resource
def load_model():
    with open('sugarcane.pkl', 'rb') as f:
        model = pickle.load(f)
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# Definir classes
classes = ['Healthy', 'Mosaic', 'RedHot', 'Rust', 'Yellow']

# Definir função de pré-processamento da imagem
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Adiciona dimensão de batch
    return image.to(device)  # Mover a imagem para o dispositivo correto

# Upload de imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagem Carregada', use_column_width=True)
    
    # Pré-processar a imagem
    image_tensor = preprocess_image(image)
    
    # Realizar a previsão
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()
        predicted_class = classes[prediction]
    
    # Exibir o resultado
    st.write(f"**Classe Predita:** {predicted_class}")

