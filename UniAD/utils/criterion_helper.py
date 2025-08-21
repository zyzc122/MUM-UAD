import torch.nn as nn


class FeatureMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        feature_rec1 = input["feature_rec"]
        # feature_rec2 = input["feature_rec2"]
        # feature_rec3 = input["feature_rec3"]
        # feature_rec2 = input["feature_rec2"]
        feature_align = input["feature_align"]
        return self.criterion_mse(feature_rec1, feature_align)
               # self.criterion_mse(feature_rec2, feature_align)+\
               # self.criterion_mse(feature_rec3, feature_align)


class SplitFeatureMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        loss = 0
        feature_rec = input["features_rec"]
        feature_align = input["features"]
        for i in range(4):
            loss = loss + self.criterion_mse(feature_rec[i], feature_align[i])
        return loss

class ImageMSELoss(nn.Module):
    """Train a decoder for visualization of reconstructed features"""

    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        image = input["image"]
        image_rec = input["image_rec"]
        return self.criterion_mse(image, image_rec)


def build_criterion(config):
    loss_dict = {}
    for i in range(len(config)):
        cfg = config[i]
        loss_name = cfg["name"]
        loss_dict[loss_name] = globals()[cfg["type"]](**cfg["kwargs"])
    return loss_dict
