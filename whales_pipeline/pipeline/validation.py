import os
import faiss
import numpy as np
import torch
import timm
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt


def get_embedding(image_path, model, shape):
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((shape, shape)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    tensor = transform(img).unsqueeze(0).to('cuda')
    with torch.no_grad():
        features = model(tensor)
    return features.cpu().numpy().flatten()


def build_faiss_index(folder_path, model, shape):
    embeddings = []
    paths = []

    for root, _, files in os.walk(folder_path):
        for file in tqdm(files, desc=f"Обработка {os.path.basename(root)}"):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                emb = get_embedding(path, model, shape)
                if emb is not None:
                    embeddings.append(emb)
                    paths.append(path)

    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    return index, paths


def search_similar(query_path, index, paths, model, shape, k=5):
    query_emb = get_embedding(query_path, model, shape)
    if query_emb is None:
        return []

    distances, indices = index.search(np.array([query_emb]).astype('float32'), k)
    return [(paths[i], distances[0][j]) for j, i in enumerate(indices[0])]


def evaluation(test_folder, index, paths, model, shape, k_values=(1, 5)):
    y_true = []
    y_pred = []
    all_scores = {k: [] for k in k_values}
    for root, _, files in os.walk(test_folder):
        for file in tqdm(files, desc="Оценка модели"):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                query_path = os.path.join(root, file)
                true_label = os.path.basename(root)

                results = search_similar(query_path, index, paths, model, shape, k=max(k_values))
                if not results:
                    continue

                retrieved_labels = [os.path.basename(os.path.dirname(p[0])) for p in results]
                for k in k_values:
                    all_scores[k].append(1 if true_label in retrieved_labels[:k] else 0)

                pred_label = retrieved_labels[0]
                y_true.append(true_label)
                y_pred.append(pred_label)

    return {
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'accuracy': accuracy_score(y_true, y_pred),
        'recall@k': {k: np.mean(all_scores[k]) for k in k_values}
    }

def evaluate_model(model, shape):
    train_index, train_paths = build_faiss_index(os.path.join('../whales_processed', "train"), model, shape)
    metrics = evaluation(
        test_folder=os.path.join('../whales_processed', "test"),
        index=train_index,
        paths=train_paths,
        model=model,
        shape=shape,
        k_values=(1,3,5)
    )
    print(metrics)


def full_evaluation(test_folder, index, paths, model, shape, k_values=(1, 5)):
    y_true = []
    y_pred = []
    all_scores = {k: [] for k in k_values}

    for root, _, files in os.walk(test_folder):
        for file in tqdm(files, desc="Оценка модели"):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                query_path = os.path.join(root, file)
                true_label = os.path.basename(root)

                results = search_similar(query_path, index, paths, model, shape, k=max(k_values))
                if not results:
                    continue

                retrieved_labels = [os.path.basename(os.path.dirname(p[0])) for p in results]
                for k in k_values:
                    all_scores[k].append(1 if true_label in retrieved_labels[:k] else 0)

                pred_label = retrieved_labels[0]
                y_true.append(true_label)
                y_pred.append(pred_label)

    print("\nКлассификационные метрики (top-1):")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("\nМетрики поиска:")
    for k in k_values:
        recall_at_k = np.mean(all_scores[k])
        print(f"Recall@{k}: {recall_at_k:.2%}")

    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_true),
                yticklabels=np.unique(y_true))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return {
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'accuracy': accuracy_score(y_true, y_pred),
        'recall@k': {k: np.mean(all_scores[k]) for k in k_values}
    }


def evaluate_final_model(model, shape):
    train_index, train_paths = build_faiss_index(os.path.join('../whales_processed', "train"), model, shape)
    metrics = full_evaluation(
        test_folder=os.path.join('../whales_processed', "test"),
        index=train_index,
        paths=train_paths,
        model=model,
        shape=shape,
        k_values=(1,3,5)
    )
    print(metrics)