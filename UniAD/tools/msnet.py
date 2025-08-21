# from typing import ForwardRef
import torch
import torch.nn as nn
from torch import Tensor
from models.initializer import initialize_from_cfg

# from UniAD_Gradient.uniad import build_position_embedding
from reverse_res34 import ConvBlock, ResBlock
import torch.nn.functional as F
from typing import Optional
from einops import rearrange


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        x = x.permute(0, 3, 1, 2)
        return x

class Merge_Attention_Block(nn.Module):
    def __init__(self,in_channels, out_channels, dim_feedforward, dropout, down_flag):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(in_channels, 8, dropout=dropout)
        self.linear1 = nn.Linear(in_channels, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, in_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_channels)
        self.activation = F.relu
        self.down_flag = down_flag
        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=out_channels,
                                            downscaling_factor=2)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, flabel, fea, pos_embed=None):
        _, c, h, w = flabel.size()

        # fea = fea.transpose(0, 1)
        fea = rearrange(
            fea, "b c h w  ->  (h w) b c", c=c
        )  # B x C X H x W

        flabel_ = rearrange(
            flabel, "b c h w  ->  (h w) b c", c=c
        )  # B x C X H x W
        # fea = fea.transpose(0, 1)
        tgt = self.self_attn(
            query=self.with_pos_embed(fea, pos_embed),
            key=flabel_,
            value=flabel_,
            attn_mask=None,
            key_padding_mask=None,
        )[0]

        tgt = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt)
        tgt = self.norm(tgt)

        tgt = rearrange(
            tgt, "(h w) b c -> b c h w", h=h
        )  # B x C X H x W

        if self.down_flag:
            tgt = self.patch_partition(tgt)

        return tgt

class Merge_Split_Block(nn.Module):
    def __init__(self, in_img_channels, in_fea_channels, split_channels, out_channels,
                 stride=1, bn=True):
        super().__init__()
        self.split_channels = split_channels
        self.res_img = ResBlock(in_img_channels, in_img_channels, bn=bn)
        self.res_fea = ResBlock(in_fea_channels, in_fea_channels, bn=bn)
        self.res_f = ResBlock(in_fea_channels, in_fea_channels, bn=bn)
        self.res_m1 = ResBlock(in_img_channels, in_img_channels, bn=bn)
        self.res_m2 = ResBlock(in_img_channels+in_fea_channels, in_fea_channels, bn=bn)
        self.conv1 = ResBlock(split_channels+in_fea_channels, out_channels)
        self.conv2 = ResBlock(split_channels+in_fea_channels, out_channels)
        self.conv3 = ResBlock(split_channels+in_fea_channels, out_channels)
        self.conv4 = ResBlock(split_channels+in_fea_channels, out_channels)
        self.conv5 = ResBlock(split_channels+in_fea_channels, out_channels)
        self.res_last = ResBlock(in_channels=out_channels*5, out_channels=out_channels*5, stride=2, bn=bn)
        # self.res_s2 = ResBlock(in_channels=(mid_channels+fea_channels)*num_branch, out_channels=out_channels*num_branch,
        #                        stride=stride, bn=bn, groups=num_branch)

    def forward(self, x, fea):

        xm = self.res_m1(x)
        fea = self.res_f(fea)
        xm = torch.cat([xm, fea], 1)
        xm = self.res_m2(xm)

        xs = torch.split(x, self.split_channels, 1)
        count = 0
        xs_c = []
        for x_i in xs:
            if count == 0:
                xs_c.append(self.conv1(torch.cat([x_i, xm], 1)))
            if count == 1:
                xs_c.append(self.conv2(torch.cat([x_i, xm], 1)))
            if count == 2:
                xs_c.append(self.conv3(torch.cat([x_i, xm], 1)))
            if count == 3:
                xs_c.append(self.conv4(torch.cat([x_i, xm], 1)))
            if count == 4:
                xs_c.append(self.conv5(torch.cat([x_i, xm], 1)))

            count = count + 1
        x = torch.cat(xs_c, 1)

        out = self.res_last(x)
        return out

class MergeBlock(nn.Module):
    def __init__(self, in_img_channels, in_fea_channels, out_channels,
                 stride=1, bn=True):
        super().__init__()
        self.res_img = ResBlock(in_img_channels, in_img_channels, bn=bn)
        self.res_fea = ResBlock(in_fea_channels, in_fea_channels, bn=bn)
        self.res_cat = ResBlock(in_img_channels+in_fea_channels, out_channels, bn=bn, stride=stride)
        # self.res_m2 = ResBlock(in_channels=2*fea_channels, out_channels=fea_channels, bn=bn)
        # self.res_s2 = ResBlock(in_channels=(mid_channels+fea_channels)*num_branch, out_channels=out_channels*num_branch,
        #                        stride=stride, bn=bn, groups=num_branch)

    def forward(self, x, fea):
        # 32 -> 8 -> 32 降维再升维
        x = self.res_img(x)
        # 32 -> 8 -> 16 降维
        fea = self.res_fea(fea)

        out = self.res_cat(torch.cat((x, fea), dim=1))

        return out

class MSBlock(nn.Module):
    def __init__(self, in_channels, mid_channels,
                 out_channels, in_fea_channels, fea_channels,
                 num_branch, stride=1, bn=True):
        super().__init__()
        self.mid_channels = mid_channels
        self.res_s1 = ResBlock(in_channels*num_branch, mid_channels*num_branch, bn=bn, groups=num_branch)
        self.res_m1 = ResBlock(in_channels=mid_channels*num_branch, out_channels=fea_channels, bn=bn)
        self.res_f = ResBlock(in_channels=in_fea_channels, out_channels=fea_channels, bn=bn)
        self.res_m2 = ResBlock(in_channels=2*fea_channels, out_channels=fea_channels, bn=bn)
        self.res_s2 = ResBlock(in_channels=(mid_channels+fea_channels)*num_branch, out_channels=out_channels*num_branch,
        stride=stride, bn=bn, groups=num_branch)

    def forward(self, x, fea):
        # 32 -> 8 -> 32 降维再升维
        x = self.res_s1(x)
        # 32 -> 8 -> 16 降维
        xm = self.res_m1(x)

        fea = self.res_f(fea)
        xm = torch.cat([xm, fea], 1)
        xm = self.res_m2(xm)

        xs = torch.split(x, self.mid_channels, 1)    
        xs_c = []
        for x_i in xs: 
            xs_c.append(torch.cat([x_i, xm], 1))
        x = torch.cat(xs_c, 1)
        x = self.res_s2(x)
        return x


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, feature_size, num_pos_feats=128):
        super().__init__()
        self.feature_size = feature_size  # H, W
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        i = torch.arange(self.feature_size[1], device=tensor.device)  # W
        j = torch.arange(self.feature_size[0], device=tensor.device)  # H
        x_emb = self.col_embed(i)  # W x C // 2
        y_emb = self.row_embed(j)  # H x C // 2
        pos = torch.cat(
            [
                torch.cat(
                    [x_emb.unsqueeze(0)] * self.feature_size[0], dim=0
                ),  # H x W x C // 2
                torch.cat(
                    [y_emb.unsqueeze(1)] * self.feature_size[1], dim=1
                ),  # H x W x C // 2
            ],
            dim=-1,
        ).flatten(
            0, 1
        )  # (H X W) X C
        return pos

def build_position_embedding(pos_embed_type, feature_size, hidden_dim):
    if pos_embed_type in ("v3", "learned"):
        pos_embed = PositionEmbeddingLearned(feature_size, hidden_dim // 2)
    else:
        raise ValueError(f"not supported {pos_embed_type}")
    return pos_embed

class MSNet(nn.Module):
    def __init__(self, num_branch=1, alpha=10):
        super(MSNet, self).__init__()

        self.conv1 = ConvBlock(num_branch, 64, 3, padding=1, stride=2,\
            bn=True, act=nn.ReLU())
        self.conv2 = ConvBlock(64, 128, 3, padding=1, stride=2, \
                               bn=True, act=nn.ReLU())
        # self.conv3 = ConvBlock(128, 256, 3, padding=1, stride=2, \
        #                        bn=True, act=nn.ReLU())
        # self.conv4 = ConvBlock(256, 512, 3, padding=1, stride=2, \
        #                        bn=True, act=nn.ReLU())
        self.attn1 = Merge_Attention_Block(in_channels=128, out_channels=256, dim_feedforward=1024, dropout=0.1, down_flag=True)
        self.attn2 = Merge_Attention_Block(in_channels=256, out_channels=512, dim_feedforward=1024, dropout=0.1, down_flag=True)
        self.attn3 = Merge_Attention_Block(in_channels=512, out_channels=512, dim_feedforward=1024, dropout=0.1, down_flag=False)
        # self.res_1 = ResBlock(in_channels=8*num_branch, out_channels=8*num_branch*2, stride=2,
        #                        bn=True, groups=num_branch)
        # self.res_2 = nn.Sequential(
        #     ConvBlock(512, 256, 3, padding=1, stride=1, bn=True, act=nn.ReLU()),
        #                            ConvBlock(256, 64, 3, padding=1, stride=1, bn=True, act=nn.ReLU()),
        #                            ConvBlock(64, 1, 3, padding=1, stride=1, bn=True, act=nn.ReLU()))

        self.res_2 = nn.Sequential(
            ConvBlock(512, 256, 3, padding=1, stride=1, bn=True, act=nn.ReLU()),
            ConvBlock(256, 128, 3, padding=1, stride=1, bn=True, act=nn.ReLU()),
            ConvBlock(128, 64, 3, padding=1, stride=1, bn=True, act=nn.ReLU()),
            ConvBlock(64, 1, 3, padding=1, stride=1, bn=True, act=nn.ReLU()))

        # self.res_d4 = ResBlock(in_channels=128*num_branch, out_channels=64*num_branch,
        #                        bn=True, groups=num_branch)
        # self.conv_d1 = nn.Conv2d(64*num_branch, num_branch, 3, padding=1,
        #                          stride=1, bias=True, groups=num_branch)
        # self.conv_d1 = ConvBlock(64*num_branch, num_branch, 3, padding=1, stride=1, \
        #                        bn=True, act=nn.ReLU(), groups=num_branch)

        self.sigmoid = nn.Sigmoid()
        # self.init_weight()
        # self.init_weights_xavier('xavier_uniform')
        self.alpha = alpha
        self.pos_embed = build_position_embedding(
            "learned", (56, 56), 128
        )
        initialize_from_cfg(self, {"method": "xavier_uniform"})

    def forward(self, flabels, fea_list):
        # flabels, fea_list = list1[0], list1[1]
        batch_size, _,_,_ = flabels.size()

        flabels_ = self.conv1(flabels)
        flabels_ = self.conv2(flabels_)
        # flabels_ = self.conv3(flabels_)
        # flabels_ = self.conv4(flabels_)
        # flabels_ = rearrange(
        #     flabels_, "b c h w -> (h w) b c"
        # )  # (H x W) x B x C
        pos_embed = self.pos_embed(flabels_)
        pos_embed = torch.cat(
            [pos_embed.unsqueeze(1)] * batch_size, dim=1
        )  # (H X W) x B x C
        fea_ = self.attn1(flabels_, fea_list[2], pos_embed)
        fea_ = self.attn2(fea_, fea_list[1])
        fea_ = self.attn3(fea_, fea_list[0])
        out = self.res_2(fea_)

        # fea_msb2 = self.msb2(fea_msb1, fea_list[1])
        # fea_msb3 = self.msb3(fea_msb2, fea_list[2])
        # out = self.alpha * (self.sigmoid(out))
        out =  self.alpha * (self.sigmoid(out) - 0.5)
        out = F.interpolate(out, (flabels.shape[2], flabels.shape[3]),
                            mode='bilinear', align_corners=False)
        return out

