import torch
import torch.nn as nn
import torch.optim as optim

class Tower(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Tower, self).__init__()
        layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            last_dim = h_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TwoTowerModel(nn.Module):
    def __init__(self, user_feature_dim, item_feature_dim, embedding_dim=128):
        super(TwoTowerModel, self).__init__()
        self.user_tower = Tower(user_feature_dim, [256, 128], embedding_dim)
        self.item_tower = Tower(item_feature_dim, [256, 128], embedding_dim)

    def forward(self, user_features, item_features):
        user_embedding = self.user_tower(user_features)
        item_embedding = self.item_tower(item_features)
        # Dot product similarity
        similarity = torch.sum(user_embedding * item_embedding, dim=1)
        return similarity

    def get_user_embedding(self, user_features):
        return self.user_tower(user_features)

    def get_item_embedding(self, item_features):
        return self.item_tower(item_features)

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device

def save_model(model, path):
    # Save as FP32 for stability
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=get_device()))
    model.to(get_device())
    # Optional: model.half() if inference only
    return model
