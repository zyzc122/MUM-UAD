import timm

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

if __name__ == '__main__':
	model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
	print(model)

