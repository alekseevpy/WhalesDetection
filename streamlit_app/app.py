import streamlit as st
from PIL import Image
import torch
import numpy as np
import model.load_model as whale_model
from pathlib import Path


@st.cache_data
def load_model():
    current_dir = Path(__file__).parent
    config_path = current_dir / "model_config.yaml"
    return whale_model.load_model(config_path)


def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    img = img.to('cpu')
    # img = img.to('cuda')
    return img


# streamlit run app.py
model = load_model()
st.title("Классификатор изображений китов на основе MegaDescriptor")

uploaded_file = st.file_uploader(
    "Выберите фото...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    key="file_uploader"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Фото загружено", use_column_width=True)

    with st.spinner("Классифицируем..."):
        try:
            input_tensor = preprocess_image(uploaded_file)
            with torch.no_grad():
                output = model(input_tensor)[0]

            st.success("Классификация завершена")
            st.subheader("Топ классов:")

            st.subheader("Все выходы модели:")
            st.write(output)

            st.bar_chart(output.numpy())

        except Exception as e:
            st.error(f"Error: {str(e)}")