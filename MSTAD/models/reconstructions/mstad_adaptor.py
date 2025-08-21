import copy
import logging
import math
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models
from einops import rearrange
from models.initializer import initialize_from_cfg
from torch import Tensor, nn

class DropBlock(nn.Module):
    def __init__(self, block_size: int, p: float = 0.5):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: Tensor) -> float:
        """计算gamma
        Args:
            x (Tensor): 输入张量
        Returns:
            Tensor: gamma
        """
        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            gamma = self.calculate_gamma(x)
            mask = torch.bernoulli(torch.ones_like(x) * gamma)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
            # x = mask_block * x
        return x

class UniAD(nn.Module):
    def __init__(
            self,
            inplanes,
            instrides,
            feature_size,
            feature_jitter,
            neighbor_mask,
            hidden_dim,
            pos_embed_type,
            save_recon,
            initializer,
            **kwargs,
    ):
        super().__init__()
        assert isinstance(inplanes, list) and len(inplanes) == 1
        assert isinstance(instrides, list) and len(instrides) == 1
        self.feature_size = feature_size
        self.num_queries = feature_size[0] * feature_size[1]
        self.feature_jitter = feature_jitter
        self.pos_embed1 = build_position_embedding(
            pos_embed_type, feature_size, hidden_dim
        )
        self.pos_embed2 = build_position_embedding(
            pos_embed_type, feature_size, hidden_dim
        )
        # self.pos_embed3 = build_position_embedding(
        #     pos_embed_type, feature_size, hidden_dim
        # )
        self.save_recon = save_recon
        self.transformer = Transformer(
            hidden_dim, feature_size, neighbor_mask, **kwargs
        )
        self.input_proj1 = nn.Linear(512, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 512)
        # self.adaptor1 = nn.Sequential(
        #     nn.Linear(inplanes[0], 512),
        #     nn.LayerNorm(512),nn.ReLU(inplace=True),
        #     nn.Linear(512, hidden_dim),
        # )
        # self.adaptor2 = nn.Sequential(
        #     nn.Linear(hidden_dim, 512),
        #     nn.LayerNorm(512),nn.ReLU(inplace=True),
        #     nn.Linear(512, inplanes[0])
        # )
        # self.input_proj1 = nn.Linear(160, hidden_dim)
        # self.output_proj = nn.Linear(hidden_dim, 160)
        # self.upsample = nn.UpsamplingBilinear2d(size=256)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=instrides[0])
        initialize_from_cfg(self, initializer)

    def add_jitter(self, feature_tokens, scale, prob):
        if random.uniform(0, 1) <= prob:
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            feature_norms = (
                    feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((num_tokens, batch_size, dim_channel)).cuda()
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens

    def forward(self, input):
        feature_align = input["feature_align"] # B x C X H x W
        # feature_tokens = rearrange(
        #     feature_align, "b c h w -> (h w) b c"
        #             )  # (H x W) x B x C
        # if self.training and self.feature_jitter:
        #     feature_tokens = self.add_jitter(
        #         feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob
        #     )
        if self.training:
            feature_mask1 = DropBlock(block_size=7, p=0.5)(feature_align)
            feature_tokens = rearrange(
                    feature_mask1, "b c h w -> (h w) b c"
                )  # (H x W) x B x C
        else:
            feature_tokens = rearrange(
                feature_align, "b c h w -> (h w) b c"
            )  # (H x W) x B x C
        # feature_tokens1 = rearrange(
        #     feature_align, "b c h w -> (h w) b c"
        # )  # (H x W) x B x C
        # feature_tokens1 = self.adaptor1(feature_tokens1)  # (H x W) x B x C
        feature_tokens1 = self.input_proj1(feature_tokens)  # (H x W) x B x C
        pos_embed1 = self.pos_embed1(feature_tokens1)  # (H x W) x C
        output_decoder, enc_list, decoder_list, tgt_list = self.transformer(
            feature_tokens1,
            pos_embed1
        )  # (H x W) x B x C
        # feature_rec_tokens1 = self.adaptor2(output_decoder)  # (H x W) x B x C
        feature_rec_tokens1 = self.output_proj(output_decoder)  # (H x W) x B x C
        feature_rec1 = rearrange(
            feature_rec_tokens1, "(h w) b c -> b c h w", h=self.feature_size[0]
        )  # B x C X H x W
        # feature_rec2 = rearrange(
        #     feature_rec_tokens2, "(h w) b c -> b c h w", h=self.feature_size[0]
        # )  # B x C X H x W

        # 反事实
        # feature_tokens_reverse = self.input_proj(feature_rec_tokens)  # (H x W) x B x C
        # pos_embed_reverse = self.pos_embed(feature_tokens_reverse)  # (H x W) x C
        # output_decoder_reverse, _ = self.transformer(
        #     feature_tokens_reverse, pos_embed_reverse
        # )  # (H x W) x B x C
        # feature_rec_tokens_reverse = self.output_proj(output_decoder_reverse)  # (H x W) x B x C
        # feature_rec_reverse = rearrange(
        #     feature_rec_tokens_reverse, "(h w) b c -> b c h w", h=self.feature_size[0]
        # )  # B x C X H x W
        pred = torch.sqrt(
            torch.sum((feature_rec1 - feature_align) ** 2, dim=1, keepdim=True)
        )  # B x 1 x H x W
        pred = self.upsample(pred)  # B x 1 x H x W

        # if not self.training and self.save_recon:
        #     clsnames = input["clsname"]
        #     filenames = input["filename"]
        #     for clsname, filename, feat_rec, preds in zip(clsnames, filenames, feature_rec1, pred):
        #         filedir, filename = os.path.split(filename)
        #         _, defename = os.path.split(filedir)
        #         filename_, _ = os.path.splitext(filename)
        #         save_dir = os.path.join('/home/scu-its-gpu-001/PycharmProjects/Saliency_MSTAD/data/MVTec-AD/{}'
        #                                 .format(self.save_recon.save_dir), clsname, defename)
        #         os.makedirs(save_dir, exist_ok=True)
        #         feature_rec_np = feat_rec.detach().cpu().numpy()
        #         np.save(os.path.join(save_dir, filename_ + ".npy"), feature_rec_np)
                # np.save(os.path.join(save_dir, filename_ + ".npy"), preds.cpu().numpy())

        return {
            "feature_rec": feature_rec1,
            "decoder_list": decoder_list,
            "encoder_list": enc_list,
            "tgt_list": tgt_list,
            "feature_align": feature_align,
            "pred": pred,
        }

class Transformer(nn.Module):
    def __init__(
            self,
            hidden_dim,
            feature_size,
            neighbor_mask,
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
        self.neighbor_mask = neighbor_mask

        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
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
        self.mse = nn.MSELoss()

    def klDivergence(self, p, q):
        p = torch.clamp(p, 1e-8, 1.0)  # 约束p的值在[1e-8, 1]范围内
        q = torch.clamp(q, 1e-8, 1.0)  # 约束q的值在[1e-8, 1]范围内
        log_ratio = torch.log(p / q)
        kl_div = torch.mean(p * log_ratio)
        kl_scaled = torch.exp(kl_div)
        return kl_scaled

    def generate_mask(self, feature_size, neighbor_size):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size
        hm, wm = neighbor_size
        mask = torch.ones(h, w, h, w)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        mask = (
            mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
                .cuda()
        )
        return mask

    def sampleing(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, src1,pos_embed1):
        _, batch_size, _ = src1.shape
        pos_embed1 = torch.cat(
            [pos_embed1.unsqueeze(1)] * batch_size, dim=1
        )  # (H X W) x B x C
        # pos_embed3 = torch.cat(
        #     [pos_embed3.unsqueeze(1)] * batch_size, dim=1
        # )  # (H X W) x B x C

        if self.neighbor_mask:
            mask = self.generate_mask(
                self.feature_size, self.neighbor_mask.neighbor_size
            )
            mask_enc = mask if self.neighbor_mask.mask[0] else None
            mask_dec1 = mask if self.neighbor_mask.mask[1] else None
            mask_dec2 = mask if self.neighbor_mask.mask[2] else None
        else:
            mask_enc = mask_dec1 = mask_dec2 = None

        output_encoder1, enc_list = self.encoder(
            src1, mask=mask_enc, pos_embed1=pos_embed1
        )  # (H X W) x B x C
        output_decoder1, decoder_list, tgt_list = self.decoder(
            output_encoder1,
            tgt_mask=mask_dec1,
            memory_mask=mask_dec2,
            pos_embed1=pos_embed1,

        )  # (H X W) x B x C
        # output_decoder2 = self.decoder(
        #     output_encoder2,
        #     tgt_mask=mask_dec1,
        #     memory_mask=mask_dec2,
        #     pos=pos_embed,
        # )  # (H X W) x B x C
        # torchvision.models.swin_s(pretrained = True)
        return output_decoder1, enc_list, decoder_list, tgt_list

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.linear_mu = nn.Linear(256, 256)
        self.linear_var = nn.Linear(256, 256)


    def forward(
            self,
            src1,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos_embed1: Optional[Tensor] = None,
    ):
        enc_list = []
        output = src1
        count = 0
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embed1=pos_embed1,
            )
            enc_list.append(output)
            count = count + 1

        if self.norm is not None:
            output = self.norm(output)

        # vae 变分思想
        # mu = self.linear_mu(output)
        # var = self.linear_var(output)

        return output, enc_list


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
            self,
            memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos_embed1: Optional[Tensor] = None,
    ):
        output = memory

        intermediate = []
        decoder_list = []
        tgt_list = []
        losses = 0
        for layer in self.layers:
            output, tgt = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos_embed1=pos_embed1,
            )
            tgt_list.append(tgt)
            decoder_list.append(output)
            # losses += loss_subspace
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output, decoder_list, tgt_list


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            hidden_dim,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.enc_head_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # self.self_gradient_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        # num_queries = 196
        self.enc_learned_embed = nn.Embedding(196, hidden_dim)  # (H x W) x C
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        # self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # self.adaptor_down_1 = nn.Linear(hidden_dim, 128)
        # self.adaptor_up_1 = nn.Linear(128, hidden_dim)
        # self.adaptor_down_2 = nn.Linear(hidden_dim, 128)
        # self.adaptor_up_2 = nn.Linear(128, hidden_dim)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        # self.self_attn_query = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.enc_norm0 = nn.LayerNorm(hidden_dim)
        self.dropout0 = nn.Dropout(dropout)



    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos_embed1: Optional[Tensor] = None,
    ):
        tgt = self.enc_learned_embed.weight
        tgt = torch.cat([tgt.unsqueeze(1)] * src.shape[1], dim=1)  # (H X W) x B x C

        src2 = self.self_attn(
            query=self.with_pos_embed(src, pos_embed1),
            key=self.with_pos_embed(src, pos_embed1),
            value=src,
            attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # encoder部分新添加的自注意力
        tgt2 = self.enc_head_attn(
            query=self.with_pos_embed(tgt, pos_embed1),
            key=self.with_pos_embed(src, pos_embed1),
            value=src,
        )[0]
        src = src + self.dropout0(tgt2)
        src = self.enc_norm0(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def forward_pre(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
            self,
            src1,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos_embed1: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src1, src_mask, src_key_padding_mask, pos_embed1)
        return self.forward_post(src1, src_mask, src_key_padding_mask, pos_embed1)

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
        # self.learned_embed = nn.Embedding(512, hidden_dim)  # (H x W) x C
        self.learned_embed = nn.Embedding(256, hidden_dim)  # (H x W) x C
        # self.dec_learned_embed_1 = nn.Embedding(196, hidden_dim)  # (H x W) x C
        # self.dec_learned_embed_2 = nn.Embedding(196, hidden_dim)  # (H x W) x C
        # self.learned_embed = nn.Embedding(num_queries, hidden_dim)  # (H x W) x C
        # nn.init.normal_(self.learned_embed.weight,mean=0, std=0.02)

        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # self.dec_head_attn_1 = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # self.dec_head_attn_2 = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # self.gradient_head_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim*2, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        # self.dec_norm_1 = nn.LayerNorm(hidden_dim)
        # self.dec_norm_2 = nn.LayerNorm(hidden_dim)
        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        # self.dec_dropout_1 = nn.Dropout(dropout)
        # self.dec_dropout_2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # self.adaptor = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.LayerNorm( hidden_dim // 2),nn.ReLU(inplace=True),
        #     nn.Linear( hidden_dim // 2, hidden_dim)
        # )

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self,
            out,
            memory,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos_embed1: Optional[Tensor] = None,
    ):
        _, batch_size, _ = memory.shape
        tgt = self.learned_embed.weight
        # dec_tgt_1 = self.dec_learned_embed_1.weight
        # dec_tgt_2 = self.dec_learned_embed_2.weight
        tgt = torch.cat([tgt.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C
        # dec_tgt_1 = torch.cat([dec_tgt_1.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C
        # dec_tgt_2 = torch.cat([dec_tgt_2.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C

        tgt_common1 = self.self_attn(
            query=self.with_pos_embed(out, pos_embed1),
            key=tgt,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt_common1 = out + self.dropout1(tgt_common1)
        tgt_common1 = self.norm1(tgt_common1)
        # tgt_common1 = self.dec_head_attn_1(
        #     query=self.with_pos_embed(dec_tgt_1, pos_embed1),
        #     key=tgt_common1,
        #     value=tgt_common1,
        #     attn_mask=tgt_mask,
        #     key_padding_mask=tgt_key_padding_mask,
        # )[0]
        # tgt_common1 = tgt_common1 + self.dec_dropout_1(tgt_common1)
        # tgt_common1 = self.dec_norm_1(tgt_common1)

        tgt_common2 = self.multihead_attn(
            query=self.with_pos_embed(memory, pos_embed1),
            key=tgt,
            value=tgt,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt_common2 = memory + self.dropout2(tgt_common2)
        tgt_common2 = self.norm2(tgt_common2)
        # tgt_common2 = self.dec_head_attn_2(
        #     query=self.with_pos_embed(dec_tgt_2, pos_embed1),
        #     key=tgt_common2,
        #     value=tgt_common2,
        #     attn_mask=memory_mask,
        #     key_padding_mask=memory_key_padding_mask,
        # )[0]
        # tgt_common2 = tgt_common2 + self.dec_dropout_2(tgt_common2)
        # tgt_common2 = self.dec_norm_2(tgt_common2)

        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_common2))))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(torch.cat((tgt_common1, tgt_common2), dim=2)))))
        tgt2 = self.norm3(tgt2)

        # dec_out = self.dec_head_attn(
        #     query=self.with_pos_embed(tgt2, pos_embed1),
        #     key=dec_tgt,
        #     value=dec_tgt,
        #     attn_mask=memory_mask,
        #     key_padding_mask=memory_key_padding_mask,
        # )[0]
        # tgt2 = tgt2 + self.dropout4(dec_out)
        # tgt2 = self.dec_norm(tgt2)

        return tgt2, tgt

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
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos_embed1: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                out,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos_embed1,
            )
        return self.forward_post(
            out,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos_embed1,
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
