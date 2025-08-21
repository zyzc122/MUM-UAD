import copy
import math
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from EWC_AD.model.initializer import initialize
from torch import Tensor, nn

from Saliency_AD.model.vis_decoder import VisDecoder


class Cross_Attn(nn.Module):
    def __init__(
        self,
        feature_size,
        feature_jitter,
            scale,prob,
        hidden_dim,
        pos_embed_type,
        save_recon,
        initializer,
        **kwargs,
    ):
        super().__init__()
        # assert isinstance(inplanes, list) and len(inplanes) == 1
        # assert isinstance(instrides, list) and len(instrides) == 1
        self.feature_size = feature_size
        self.scale = scale
        self.prob = prob
        self.num_queries = feature_size[0] * feature_size[1]
        self.feature_jitter = feature_jitter
        self.pos_embed = build_position_embedding(
            pos_embed_type, feature_size, hidden_dim
        )
        self.save_recon = save_recon

        self.transformer = Transformer(
            hidden_dim, feature_size, **kwargs
        )
        self.input_proj = nn.Linear(272, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 3)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample([224,224])

        initialize(self, initializer)
    # 加噪
    def add_jitter(self, feature_tokens, scale, prob):
        if random.uniform(0, 1) <= prob:
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            # 生成同一维度大小的噪声
            jitter = torch.randn((num_tokens, batch_size, dim_channel)).cuda()
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens

    def forward(self, flabel_feature, img_feature_list, tgt_list):
        # feature_align = input["feature_align"]  # B x C X H x W
        flabel_token = rearrange(
            flabel_feature, "b c h w -> (h w) b c"
        )  # (H x W) x B x C
        # if self.training and self.feature_jitter:
        #     flabel_token = self.add_jitter(
        #         flabel_token, self.scale, self.prob
        #     )
        # 通过线性层去减小C的维度
        flabel_tokens = self.input_proj(flabel_token)  # (H x W) x B x C
        pos_embed = self.pos_embed(flabel_tokens)  # (H x W) x C
        atten_feature = self.transformer(
            flabel_tokens, img_feature_list, tgt_list, pos_embed,
        )  # (H x W) x B x C
        atten_feature = self.output_proj(atten_feature)  # (H x W) x B x C
        atten_feature = rearrange(
            atten_feature, "(h w) b c -> b c h w", h=self.feature_size[0]
        )  # B x C X H x W

        # atten_feature = 10 * (self.sigmoid(atten_feature) - 0.5)
        atten_feature = 10 * (self.sigmoid(atten_feature))
        mask = self.upsample(atten_feature)
        # if not self.training and self.save_recon:
        #     clsnames = input["clsname"]
        #     filenames = input["filename"]
        #     for clsname, filename, feat_rec in zip(clsnames, filenames, feature_rec):
        #         filedir, filename = os.path.split(filename)
        #         _, defename = os.path.split(filedir)
        #         filename_, _ = os.path.splitext(filename)
        #         save_dir = os.path.join(self.save_recon.save_dir, clsname, defename)
        #         os.makedirs(save_dir, exist_ok=True)
        #         feature_rec_np = feat_rec.detach().cpu().numpy()
        #         np.save(os.path.join(save_dir, filename_ + ".npy"), feature_rec_np)
        return mask


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        feature_size,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()
        self.feature_size = feature_size
        # 初始化编码层的结构
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            feature_size,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self.hidden_dim = hidden_dim
        self.nhead = nhead

    def forward(self, flabel_feature, img_feature_list, tgt_list, pos_embed):
        _, batch_size, _ = flabel_feature.shape
        pos_embed = torch.cat(
            [pos_embed.unsqueeze(1)] * batch_size, dim=1
        )  # (H X W) x B x C
        mask_dec1 = mask_dec2 = None
        output_decoder = self.decoder(
            flabel_feature,
            img_feature_list,
            tgt_list,
            tgt_mask=mask_dec1,
            memory_mask=mask_dec2,
            pos=pos_embed,
        )  # (H X W) x B x C

        return output_decoder

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        flabel_feature,
        img_feature_list,
        tgt_list,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # 将out初始化为 encoder的输出，即memory
        output = flabel_feature
        memory = flabel_feature
        count = 0
        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                img_feature_list[count],
                tgt_list[count],
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
            )
            count = count + 1

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        feature_size,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        num_queries = feature_size[0] * feature_size[1]
        self.learned_embed = nn.Embedding(num_queries, hidden_dim)  # (H x W) x C

        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.linear3 = nn.Linear(1280, 256)
        self.adaptor1 = nn.Linear(64,64)
        self.adaptor2= nn.Linear(64,64)
        self.adaptor3= nn.Linear(64,64)
        self.adaptor4 = nn.Linear(64,64)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        out,
        memory,
        img_feature,
        tgt_embedding,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        hw, batch_size, _ = out.shape
        # tgt = self.learned_embed.weight
        # tgt = torch.cat([tgt.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C

        # tgt1 = self.multihead_attn(
        #     query=self.with_pos_embed(out, pos),
        #     key=self.with_pos_embed(out, pos),
        #     value=out,
        #     attn_mask=tgt_mask,
        #     key_padding_mask=tgt_key_padding_mask,
        # )[0]
        # tgt1 = out + self.dropout2(tgt1)
        # out = self.norm2(tgt1)

        # xs = torch.split(out, 64, 2)
        # xs_c = []
        # xs_c.append(self.adaptor1(xs[0]))
        # xs_c.append(self.adaptor2(xs[1]))
        # xs_c.append(self.adaptor3(xs[2]))
        # xs_c.append(self.adaptor4(xs[3]))
        # # for x_i in xs:
        # #     xs_c.append(torch.cat([x_i, out], 2))
        # out = torch.cat(xs_c, 2)
        # out = self.linear3(x)

        tgt1 = self.self_attn(
            query=self.with_pos_embed(img_feature, pos),
            key=self.with_pos_embed(out, pos),
            value=out,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt1 = out + self.dropout1(tgt1)
        tgt1 = self.norm1(tgt1)

        # tgt2 = self.multihead_attn(
        #     query=self.with_pos_embed(tgt1, pos),
        #     # key=self.with_pos_embed(tgt_embedding, pos),
        #     key=tgt_embedding,
        #     value=tgt_embedding,
        #     attn_mask=memory_mask,
        #     key_padding_mask=memory_key_padding_mask,
        # )[0]
        # tgt2 = tgt1 + self.dropout2(tgt2)
        # tgt2 = self.norm2(tgt2)


        # tgt1 = self.self_attn(
        #     query=self.with_pos_embed(memory, pos),
        #     key=self.with_pos_embed(img_feature, pos),
        #     value=img_feature,
        #     attn_mask=tgt_mask,
        #     key_padding_mask=tgt_key_padding_mask,
        # )[0]
        # tgt1 = memory + self.dropout1(tgt1)
        # tgt1 = self.norm1(tgt1)
        #
        # tgt2 = self.multihead_attn(
        #     query=self.with_pos_embed(tgt1, pos),
        #     key=self.with_pos_embed(out, pos),
        #     value=out,
        #     attn_mask=memory_mask,
        #     key_padding_mask=memory_key_padding_mask,
        # )[0]
        # tgt2 = tgt1 + self.dropout2(tgt2)
        # tgt2 = self.norm2(tgt2)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt1))))
        # tgt = self.linear2(self.dropout(self.activation(self.linear1(torch.cat((tgt1, tgt2), dim=2)))))
        tgt = tgt + self.dropout3(tgt1)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        out,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        _, batch_size, _ = memory.shape
        tgt = self.learned_embed.weight
        tgt = torch.cat([tgt.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C

        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(out, pos),
            value=out,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        out,
        memory,
        img_feature,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                out,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
            )
        return self.forward_post(
            out,
            memory,
            img_feature,
            tgt,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        feature_size,
        num_pos_feats=128,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        not_mask = torch.ones((self.feature_size[0], self.feature_size[1]))  # H x W
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)  # (H X W) X C
        return pos.to(tensor.device)


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
    if pos_embed_type in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        pos_embed = PositionEmbeddingSine(feature_size, hidden_dim // 2, normalize=True)
    elif pos_embed_type in ("v3", "learned"):
        pos_embed = PositionEmbeddingLearned(feature_size, hidden_dim // 2)
    else:
        raise ValueError(f"not supported {pos_embed_type}")
    return pos_embed
