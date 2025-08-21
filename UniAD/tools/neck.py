import torch
import torch.nn as nn

class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()

    def forward(self, input):
        features = input["features"]

        feature_list = []
        upsample_ratio_list = [0.125, 0.25, 0.5, 1]
        # resize & concatenate
        for i in range(len(features)):
            feature_resize = nn.UpsamplingBilinear2d(scale_factor=upsample_ratio_list[i])(features[i])
            feature_list.append(feature_resize)

        feature_align = torch.cat(feature_list, dim=1)

        return feature_align
