import timm
import torch
import torch.nn as nn
import yaml


def load_model(yaml_path):
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    config = config["load_model"]

    if config["compress_model"]:
        model = load_compressed_embedding_model(
            model_name=config["model_name"],
            embedding_size=config["compressed_shape"],
            device="cpu",
            # device="cuda",
            weights_path=config["weights_path"],
        )
    else:
        model = load_base_embedding_model(
            model_name=config["model_name"],
            device="cpu",
            # device="cuda",
            weights_path=config["weights_path"],
        )

    return model


def load_compressed_embedding_model(
    model_name="hf-hub:BVRA/MegaDescriptor-S-224",
    embedding_size=512,
    device="cuda",
    weights_path="./best_weights.pth",
):

    # todo надо разобраться, как EmbeddingModel научиться загружать из модуля со streamlit
    class EmbeddingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = create_base_model(model_name, device, weights_path)
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
    model_name="hf-hub:BVRA/MegaDescriptor-S-224",
    device="cuda",
    weights_path="./best_weights.pth",
):
    return create_base_model(model_name, device, weights_path).to(device)


def create_base_model(
    model_name="hf-hub:BVRA/MegaDescriptor-S-224",
    device="cuda",
    weights_path="./best_weights.pth",
):
    model = timm.create_model(model_name, pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    return model
