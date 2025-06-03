import streamlit as st
from PIL import Image
import torch
import numpy as np
import model.load_model as whale_model
from pathlib import Path
import faiss
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

@st.cache_data
def load_model():
    current_dir = Path(__file__).parent
    config_path = current_dir / "model_config.yaml"
    print('model is loaded')
    return whale_model.load_model(config_path)


@st.cache_data
def load_index():
    current_dir = Path(__file__).parent
    index_path = str(current_dir / 'whales_faiss_index.pkl.index')
    meta_path = str(current_dir / 'whales_faiss_index.pkl.json')
    index = faiss.read_index(index_path)

    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    data = {
        'index': index,
        'paths': metadata['paths'],
        'class_names': metadata['class_names'],
        'class_to_idx': metadata['class_to_idx'],
        'idx_to_class': metadata['idx_to_class'],
        'model_name': metadata['model_name'],
        'image_size': metadata['image_size']
    }

    print('data is loaded')
    return data

def search_similar_images(embedding, index, k=5):
    distances, indices = index['index'].search(np.array([embedding]).astype('float32'), k)

    results = []
    for i, idx in enumerate(indices[0]):
        path = index['paths'][idx]
        class_name = index['class_names'][idx]
        distance = distances[0][i]
        results.append({
            'path': path,
            'class': class_name,
            'distance': float(distance)
            # 'image': Image.open(path)
        })

    return results


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
index = load_index()
st.title("Классификатор изображений китов на основе MegaDescriptor")

uploaded_file = st.file_uploader(
    "Выберите фото...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    key="file_uploader"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Фото загружено", use_container_width=True)

    with st.spinner("Классифицируем..."):
        try:
            input_tensor = preprocess_image(uploaded_file)
            with torch.no_grad():
                output = model(input_tensor)[0]
            embedding = output.cpu().numpy().flatten()
            search_result = search_similar_images(embedding, index)

            st.success("Классификация завершена")
            st.subheader("Топ 5 похожих классов:")
            seen_classes = set()
            for res in search_result:
                if res['class'] not in seen_classes:
                    st.write(f"- {res['class']} (distance: {res['distance']:.2f})")
                    seen_classes.add(res['class'])
                    if len(seen_classes) == 5:
                        break
        except Exception as e:
            st.error(f"Error: {str(e)}")