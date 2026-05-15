import math

import torch
import torch.nn as nn
from model.audio.torchvggish import vggish
from timm.models.layers import DropPath, trunc_normal_

from model.visual.sam2.modeling.position_encoding import PositionEmbeddingSine


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, norm_act=nn.BatchNorm2d, conv_layer=nn.Conv2d):
        super().__init__()
        self.proj = nn.Sequential(
            conv_layer(dim_in, proj_dim, kernel_size=1),
            norm_act(proj_dim),
            conv_layer(proj_dim, proj_dim, kernel_size=1),
        )

    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), p=2, dim=1)

class AuralFuser(torch.nn.Module):
    """Fuses VGGish audio with SAM2 FPN maps via patch embeds, fusion blocks, and projection heads."""

    def __init__(self, hyp_param):
        self.hyp_param = hyp_param
        super().__init__()
        self.vgg = vggish.VGGish(self.hyp_param.audio)
        if not getattr(self.hyp_param, "train_vggish", False):
            for p in self.vgg.parameters():
                p.requires_grad = False

        self.position_encoding_func = PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None,
                                                            temperature=10000)

        # Populated in main.py / inference.py via Hydra compose('auralfuser/architecture.yaml') → hyp_param.aural_fuser
        if not hasattr(self.hyp_param, "aural_fuser") or self.hyp_param.aural_fuser is None:
            raise ValueError(
                "hyp_param.aural_fuser is missing; load it with Hydra compose before constructing AuralFuser."
            )
        arch_cfg = self.hyp_param.aural_fuser

        _patch_cfgs = [tuple(i) for i in arch_cfg["patch_cfgs"]]
        _f_depths = arch_cfg["f_depths"]
        _block_kw = dict(arch_cfg["block_kw"])
        _block_kw["norm_layer"] = nn.LayerNorm
        _one_d_kw = dict(arch_cfg["one_d_kw"])
        _one_d_kw["norm_layer"] = nn.LayerNorm
        self.patch_embeds = nn.ModuleList(
            nn.Conv2d(256, 256, kernel_size=k, stride=s) for k, s in _patch_cfgs
        )

        self.f_blocks = nn.ModuleList(
            nn.ModuleList([Block(**_block_kw) for _ in range(n)]) for n in _f_depths
        )

        self.a_blocks = nn.ModuleList(
            nn.ModuleList([OneDBlock(**_one_d_kw) for _ in range(3)]) for _ in range(3)
        )

        self.fusion_modules = nn.ModuleList(
            AudioVisualFusionModule(in_channels=256, mode='dot') for _ in range(3)
        )
        self.smooth_convs = nn.ModuleList(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0) for _ in range(2)
        )

        self.train_proj_v1 = ProjectionHead(dim_in=256, proj_dim=128)

        self.train_proj_a1 = ProjectionHead(dim_in=256, norm_act=nn.BatchNorm1d, conv_layer=nn.Conv1d, proj_dim=128)

    @staticmethod
    def positionalencoding1d(d_model, length):
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe

    def forward(self, feature_dicts, spect=None):
        image_embed_shape = [self.hyp_param.image_embedding_size] * 2
        H, W = image_embed_shape[0], image_embed_shape[1]
        d = torch.cat(
            [
                self.vgg(spect[:, 0, ...].unsqueeze(1)),
                self.vgg(spect[:, 1, ...].unsqueeze(1)),
            ],
            dim=-1,
        )
        length = d.shape[-1]
        fix_audio_pos = self.positionalencoding1d(length, 1).squeeze().to(spect.device)
        fpn = list(feature_dicts["backbone_fpn"])
        patch_embeds = list(self.patch_embeds)
        f_blocks = list(self.f_blocks)
        a_blocks = list(self.a_blocks)
        tpavi = list(self.fusion_modules)
        smooths = [None, self.smooth_convs[0], self.smooth_convs[1]]

        feats = [None, None, None]
        d_outputs = []

        for i in range(3):
            x = fpn[i]
            x = patch_embeds[i](x)
            x_pos = self.position_encoding_func(x)
            x = x.flatten(2).permute(0, 2, 1)
            x_pos = x_pos.flatten(2).permute(0, 2, 1)

            if i == 0:
                x = x + x_pos
                d = d + fix_audio_pos
            else:
                x = x + feats[i - 1]
                x = smooths[i](
                    x.permute(0, 2, 1).reshape(x.shape[0], 256, H, W)
                ).flatten(2).permute(0, 2, 1)
                x = x + x_pos
                d = d + fix_audio_pos

            for blks in f_blocks[i]:
                x = blks(x, H, W, x_pos)
            for blks in a_blocks[i]:
                d = blks(d, fix_audio_pos)

            x = x + x_pos
            d = d + fix_audio_pos
            x, d_out, _, _ = tpavi[i](x, H, W, x_pos, d, length)
            d = d_out
            feats[i] = x
            d_outputs.append(d_out)

        a, b, c = feats
        d1, d2, d3 = d_outputs

        feature_residual = [a, b, c]
        audio_out = [d1, d2, d3]

        proj_feature_out = [
            [
                self.train_proj_v1(a.permute(0, 2, 1).reshape(-1, 256, *image_embed_shape)),
                self.train_proj_v1(b.permute(0, 2, 1).reshape(-1, 256, *image_embed_shape)),
                self.train_proj_v1(c.permute(0, 2, 1).reshape(-1, 256, *image_embed_shape)),
            ],
            [
                self.train_proj_a1(d1.unsqueeze(-1)),
                self.train_proj_a1(d2.unsqueeze(-1)),
                self.train_proj_a1(d3.unsqueeze(-1)),
            ],
        ]

        return feature_residual, audio_out, proj_feature_out


class AudioVisualFusionModule(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='dot',
                 dimension=3):
        super().__init__()
        assert mode == 'dot'
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        self.align_channel = nn.Conv1d(256, in_channels, kernel_size=1)
        self.align_channel_back = nn.Conv1d(in_channels, 128, kernel_size=1)

        self.norm_layer = nn.LayerNorm(in_channels)

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        self.W_z = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)

        self.W_z2 = nn.Sequential(
            nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
            nn.BatchNorm1d(self.in_channels)
        )
        nn.init.constant_(self.W_z2[1].weight, 0)
        nn.init.constant_(self.W_z2[1].bias, 0)
        self.norm_layer2 = nn.LayerNorm(self.in_channels)

        self.q_frame = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.k_frame = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.v_frame = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        self.q_audio = nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        self.k_audio = nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)
        self.v_audio = nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1)

    def forward(self, frame, H_x, W_x, tmp1, audio, tmp2):
        frame = frame.permute(0, 2, 1)
        frame = frame.reshape(frame.shape[0], frame.shape[1], H_x, W_x)
        frame = frame.unsqueeze(2)
        audio = self.align_channel(audio.unsqueeze(-1))

        batch_size, _ = frame.size(0), frame.size(1)
        q_frame = self.q_frame(frame).reshape(1, -1, self.inter_channels)
        k_frame = self.k_frame(frame).reshape(1, -1, self.inter_channels)
        v_frame = self.v_frame(frame).reshape(1, -1, self.inter_channels)
        q_audio = self.q_audio(audio).reshape(1, -1, self.inter_channels)
        k_audio = self.k_audio(audio).reshape(1, -1, self.inter_channels)
        v_audio = self.v_audio(audio).reshape(1, -1, self.inter_channels)
        f = torch.matmul(q_frame, k_audio.mT)
        f_normalise = f / f.size(1)

        frame_attn = torch.matmul(f_normalise, v_audio)

        frame_attn = frame_attn.permute(0, 2, 1).contiguous()
        frame_attn = frame_attn.view(batch_size, self.inter_channels, *frame.size()[2:])
        frame_attn = self.W_z(frame_attn)
        frame = frame_attn + frame

        frame = frame.permute(0, 2, 3, 4, 1)
        frame = self.norm_layer(frame)
        frame = frame.permute(0, 4, 1, 2, 3)
        frame = frame.squeeze().flatten(start_dim=2).permute(0, 2, 1)

        a = torch.matmul(q_audio, k_frame.mT)
        a_normalise = a / a.size(-1)

        audio_attn = torch.matmul(a_normalise, v_frame)
        audio_attn = audio_attn.permute(0, 2, 1).contiguous()

        audio_attn = audio_attn.view(batch_size, self.inter_channels).unsqueeze(-1)
        audio_attn = self.W_z2(audio_attn)

        audio = audio_attn + audio

        audio = self.norm_layer2(audio.squeeze()).squeeze()

        return frame, audio, frame_attn, audio_attn


class OneDBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = OneDAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = OneDMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, _pos):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class OneDAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x.unsqueeze(0)

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.squeeze()
        return x


class OneDMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear

        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, _pos):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear

        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x
