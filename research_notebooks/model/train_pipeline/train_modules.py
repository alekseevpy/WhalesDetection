import os
import random
from collections import defaultdict

import albumentations as A
import numpy as np
import timm
import torch
import torch.nn as nn
import validation
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_metric_learning import losses, miners
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm


class BalancedDataGenerator(Dataset):
    def __init__(
        self,
        dataset,
        class_counts,
        max_count,
        transform_aggressive,
        transform_moderate,
        transform_basic,
    ):
        self.dataset = dataset
        self.class_counts = class_counts
        self.max_count = max_count
        self.transform_aggressive = transform_aggressive
        self.transform_moderate = transform_moderate
        self.transform_basic = transform_basic

        self.class_to_images = defaultdict(list)
        for img_path, label in dataset.imgs:
            self.class_to_images[label].append(img_path)

    def __len__(self):
        return self.max_count * len(self.class_counts)

    def __getitem__(self, idx):
        class_idx = idx // self.max_count
        image_idx = idx % self.max_count

        label = list(self.class_counts.keys())[class_idx]
        images = self.class_to_images[label]

        if image_idx >= len(images):
            img_path = random.choice(images)
            image = Image.open(img_path).convert("RGB")
            if self.class_counts[label] < 0.25 * self.max_count:
                transform = self.transform_aggressive
            elif self.class_counts[label] < 0.75 * self.max_count:
                transform = self.transform_moderate
            else:
                transform = self.transform_basic

            image_np = np.array(image)
            augmented = transform(image=image_np)
            image = augmented["image"]
        else:
            img_path = images[image_idx]
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)
            augmented = self.transform_basic(image=image_np)
            image = augmented["image"]

        return image, label


def set_transforms(shape):
    transform_aggressive = A.Compose(
        [
            A.RandomRotate90(p=0.9),
            # A.HorizontalFlip(p=0.6),
            # A.VerticalFlip(p=0.2),
            A.Resize(shape, shape),
            # A.CoarseDropout(max_holes=5, max_height=20, max_width=20, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    transform_moderate = A.Compose(
        [
            A.RandomRotate90(p=0.45),
            # A.HorizontalFlip(p=0.3),
            A.Resize(shape, shape),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    transform_basic = A.Compose(
        [
            A.Resize(shape, shape),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    return transform_aggressive, transform_moderate, transform_basic


def get_class_counts(dataset):
    counts = defaultdict(int)
    for _, label in dataset.samples:
        counts[label] += 1
    return dict(counts)


def load_compressed_embedding_model(
    model_name="hf-hub:BVRA/MegaDescriptor-S-224",
    embedding_size=512,
    device="cuda",
):
    class EmbeddingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(model_name, pretrained=True)
            self.projection = nn.Linear(
                self.backbone.num_features, embedding_size
            )

        def forward(self, x):
            features = self.backbone(x)
            embeddings = self.projection(features)
            return nn.functional.normalize(embeddings, p=2, dim=1)

    model = EmbeddingModel().to(device)
    return model


def load_base_embedding_model(
    model_name="hf-hub:BVRA/MegaDescriptor-S-224", device="cuda"
):

    model = timm.create_model(model_name, pretrained=True).to(device)
    return model


def prepare_training_modules(yaml_path):
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    config = config["prepare_modules"]

    train_dataset = ImageFolder(root="../whales_processed/train")
    train_counts = get_class_counts(train_dataset)

    transform_aggressive, transform_moderate, transform_basic = set_transforms(
        config["shape"]
    )
    train_generator = BalancedDataGenerator(
        dataset=train_dataset,
        class_counts=train_counts,
        max_count=max(train_counts.values()),
        transform_aggressive=transform_aggressive,
        transform_moderate=transform_moderate,
        transform_basic=transform_basic,
    )

    train_loader = DataLoader(
        train_generator,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    if config["compress_model"]:
        model = load_compressed_embedding_model(
            model_name=config["model_name"],
            embedding_size=config["compressed_shape"],
            device="cuda",
        )
    else:
        model = load_base_embedding_model(
            model_name=config["model_name"], device="cuda"
        )

    return train_loader, model, config["shape"]


def train_model(
    model,
    train_loader,
    shape,
    epochs=10,
    lr=1e-4,
    device="cuda",
    weights_dir="best_model",
    eval_step=1,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_func = losses.TripletMarginLoss(margin=0.2)
    miner = miners.TripletMarginMiner(margin=0.1)

    os.makedirs(weights_dir, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        skipped_batches = 0

        batch_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False
        )

        for images, labels in batch_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            embeddings = model(images)
            triplets = miner(embeddings, labels)

            # if isinstance(triplets, tuple) and len(triplets) > 0 and len(triplets[0]) == 0:
            #    skipped_batches += 1
            #    batch_bar.set_postfix(loss="skip (no triplets)", skipped=skipped_batches)
            #    continue

            loss = loss_func(embeddings, labels, triplets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            batch_bar.set_postfix(
                loss=loss.item(),
                avg_loss=total_loss / (batch_bar.n + 1 - skipped_batches),
                skipped=skipped_batches,
            )

        batch_bar.close()

        avg_loss = (
            total_loss / (len(train_loader) - skipped_batches)
            if (len(train_loader) - skipped_batches) > 0
            else 0
        )
        print(
            f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Skipped: {skipped_batches}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(),
                os.path.join(weights_dir, "best_weights.pth"),
            )
            print(f"New best weights saved with loss {best_loss:.4f}")

        if (epoch + 1) % eval_step == 0:
            print("start validation")
            validation.evaluate_model(model, shape)

    model.load_state_dict(
        torch.load(os.path.join(weights_dir, "best_weights.pth"))
    )

    return model
