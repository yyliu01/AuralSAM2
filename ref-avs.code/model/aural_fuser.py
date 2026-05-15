import torch
import torch.nn as nn
from model.audio.torchvggish import vggish
from timm.models.layers import DropPath, trunc_normal_
import math

from model.visual.sam2.modeling.position_encoding import PositionEmbeddingSine


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, norm_act=nn.BatchNorm2d, conv_layer=nn.Conv2d):
        super(ProjectionHead, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim_in, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), p=2, dim=1)


class AuralFuser(torch.nn.Module):
    """Fuses VGGish audio, RoBERTa text, and SAM2 FPN maps via patch embeds, fusion blocks, and projection heads."""

    def __init__(self, hyp_param):
        self.hyp_param = hyp_param
        super().__init__()
        self.vgg = vggish.VGGish(self.hyp_param.audio)
        if not getattr(self.hyp_param, "train_vggish", False):
            for p in self.vgg.parameters():
                p.requires_grad = False

        self.position_encoding_func = PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None,
                                                            temperature=10000)

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
            TPAVIModuleDIY(in_channels=256, mode='dot') for _ in range(3)
        )
        self.smooth_convs = nn.ModuleList(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0) for _ in range(2)
        )

        self.train_proj_v1 = ProjectionHead(dim_in=256, proj_dim=128)
        self.train_proj_a1 = ProjectionHead(dim_in=256, proj_dim=128)

        self.text_proj = nn.Sequential(
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Linear(1024, 256),
        )

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

    def forward(self, feature_dicts, spect=None, text=None):
        image_embed_shape = [self.hyp_param.image_embedding_size] * 2
        H, W = image_embed_shape[0], image_embed_shape[1]
        d = torch.cat(
            [
                self.vgg(spect[:, 0, ...].unsqueeze(1)),
                self.vgg(spect[:, 1, ...].unsqueeze(1)),
            ],
            dim=-1,
        )
        text = self.text_proj(text)
        d = torch.cat([d, text.squeeze()])

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
        vis_attn_feats = []

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
            x, d_out, x_attn, _ = tpavi[i](x, H, W, x_pos, d, length)
            d = d_out
            feats[i] = x
            d_outputs.append(d_out)
            vis_attn_feats.append(x_attn)

        a, b, c = feats
        d1, d2, d3 = d_outputs
        a_attn, b_attn, c_attn = vis_attn_feats

        feature_residual = [a, b, c]
        audio_out = [d1, d2, d3]

        proj_feature_out = [
            [
                self.train_proj_v1(a_attn.flatten(start_dim=2).permute(0, 2, 1)).reshape(
                    -1, *image_embed_shape, 128
                ).permute(0, 3, 1, 2),
                self.train_proj_v1(b_attn.flatten(start_dim=2).permute(0, 2, 1)).reshape(
                    -1, *image_embed_shape, 128
                ).permute(0, 3, 1, 2),
                self.train_proj_v1(c_attn.flatten(start_dim=2).permute(0, 2, 1)).reshape(
                    -1, *image_embed_shape, 128
                ).permute(0, 3, 1, 2),
            ],
            [
                self.train_proj_a1(d1[:10]).unsqueeze(-1),
                self.train_proj_a1(d2[:10]).unsqueeze(-1),
                self.train_proj_a1(d3[:10]).unsqueeze(-1),
            ],
        ]

        return feature_residual, audio_out, proj_feature_out


class TPAVIModuleDIY(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='dot',
                 dimension=3):
        """
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(TPAVIModuleDIY, self).__init__()
        assert mode == 'dot', print('... following original paper.')
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

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
        """
        args:
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
            audio: (N, T, C)
        """
        frame = frame.permute(0, 2, 1)
        frame = frame.reshape(frame.shape[0], frame.shape[1], H_x, W_x)

        frame = frame.unsqueeze(2)
        audio = self.align_channel(audio.unsqueeze(-1))


        batch_size = frame.size(0)
        audio_batch_size = audio.size(0)
        q_frame = self.q_frame(frame).reshape(1, -1, self.inter_channels)  # [bs, 4096, 128]
        k_frame = self.k_frame(frame).reshape(1, -1, self.inter_channels)  # [bs, 4096, 128]
        v_frame = self.v_frame(frame).reshape(1, -1, self.inter_channels)  # [bs, 4096, 128]

        q_audio = self.q_audio(audio).reshape(1, -1, self.inter_channels)  # [bs, 1, 128]
        k_audio = self.k_audio(audio).reshape(1, -1, self.inter_channels)  # [bs, 1, 128]
        v_audio = self.v_audio(audio).reshape(1, -1, self.inter_channels)  # [bs, 1, 128]

        f = torch.matmul(q_frame, k_audio.mT)  # [bs, 4096, 1]
        f_normalise = f / f.size(1)  # [bs, THW, THW]

        frame_attn = torch.matmul(f_normalise, v_audio)  # [bs, THW, C]

        frame_attn = frame_attn.permute(0, 2, 1).contiguous()  # [bs, C, THW]
        frame_attn = frame_attn.view(batch_size, self.inter_channels, *frame.size()[2:])  #
        frame_attn = self.W_z(frame_attn)  # [bs, C, T, H, W]
        frame = frame_attn + frame  # # [bs, C, T, H, W]

        frame = frame.permute(0, 2, 3, 4, 1)  # [bs, T, H, W, C]
        frame = self.norm_layer(frame)
        frame = frame.permute(0, 4, 1, 2, 3)  # [bs, C, T, H, W]
        frame = frame.squeeze().flatten(start_dim=2).permute(0, 2, 1)

        a = torch.matmul(q_audio, k_frame.mT)  # [bs, THW, THW]
        a_normalise = a / a.size(-1)

        audio_attn = torch.matmul(a_normalise, v_frame)
        audio_attn = audio_attn.permute(0, 2, 1).contiguous()  # [bs, C, THW]

        audio_attn = audio_attn.view(audio_batch_size, self.inter_channels).unsqueeze(-1)
        audio_attn = self.W_z2(audio_attn)  # [bs, C, T, H, W]

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

    def forward(self, x, pos):
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


class TwoWayBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1_f = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.attn = TwoWayCrossAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_f = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_f = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)
        self.mlp_a = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

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

    def forward(self, x_f, H_f, W_f, x_f_pos, x_a, H_a, W_a, x_a_pos):
        x_f_1, x_a_1 = self.attn(self.norm1_f(x_f + x_f_pos), H_f, W_f, self.norm1_a(x_a + x_a_pos), H_a, W_a)
        x_f, x_a = x_f + self.drop_path(x_f_1), x_a + self.drop_path(x_a_1)

        x_f_2, x_a_2 = self.mlp_f(self.norm2_f(x_f), H_f, W_f), self.mlp_a(self.norm2_a(x_a), H_a, W_a)
        x_f, x_a = x_f + self.drop_path(x_f_2), x_a + self.drop_path(x_a_2)
        return x_f, x_a


class TwoWayCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.linear = linear
        self.sr_ratio = sr_ratio
        for i in ['frame', 'audio']:
            setattr(self, i + '_q', nn.Linear(dim, dim, bias=qkv_bias))
            setattr(self, i + '_kv', nn.Linear(dim, dim, bias=qkv_bias))
            setattr(self, i + '_attn_drop', nn.Dropout(attn_drop))
            setattr(self, i + '_proj', nn.Linear(dim, dim, bias=qkv_bias))
            setattr(self, i + '_proj_drop', nn.Dropout(proj_drop))
            if not linear:
                if sr_ratio > 1:
                    setattr(self, i + '_sr', nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio))
                    setattr(self, i + '_norm', nn.LayerNorm(dim))
            else:
                setattr(self, i + '_pool', nn.AdaptiveAvgPool2d(7))
                setattr(self, i + '_sr', nn.Conv2d(dim, dim, kernel_size=1, stride=1))
                setattr(self, i + '_norm', nn.LayerNorm(dim))
                setattr(self, i + '_act', nn.GELU())


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

    def forward(self, x_f, H_f, W_f, x_a, H_a, W_a):
        B_f, N_f, C_f = x_f.shape
        B_a, N_a, C_a = x_a.shape
        q_f = self.frame_q(x_f).reshape(B_f, N_f, self.num_heads, C_f // self.num_heads).permute(0, 2, 1, 3)
        q_a = self.audio_q(x_a).reshape(B_a, N_a, self.num_heads, C_a // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_f = x_f.permute(0, 2, 1).reshape(B_f, C_f, H_f, W_f)
                x_f = self.frame_sr(x_f).reshape(B_f, C_f, -1).permute(0, 2, 1)
                x_f = self.frame_norm(x_f)
                kv_f = self.frame_kv(x_f).reshape(B_f, -1, 2, self.num_heads, C_f // self.num_heads).permute(2, 0, 3, 1,
                                                                                                             4)

                x_a = x_a.permute(0, 2, 1).reshape(B_a, C_a, H_a, W_a)
                x_a = self.audio_sr(x_a).reshape(B_a, C_a, -1).permute(0, 2, 1)
                x_a = self.audio_norm(x_a)
                kv_a = self.audio_kv(x_a).reshape(B_a, -1, 2, self.num_heads, C_f // self.num_heads).permute(2, 0, 3, 1,
                                                                                                             4)

            else:
                kv_f = self.frame_kv(x_f).reshape(B_f, -1, 2, self.num_heads, C_f // self.num_heads).permute(2, 0, 3, 1,
                                                                                                             4)
                kv_a = self.kv(x_a).reshape(B_a, -1, 2, self.num_heads, C_a // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            raise NotImplementedError

        k_f, v_f = kv_f[0], kv_f[1]
        k_a, v_a = kv_a[0], kv_a[1]

        attn_a = (q_a @ k_f.transpose(-2, -1)) * self.scale
        attn_a = attn_a.softmax(dim=-1)
        attn_a = self.audio_attn_drop(attn_a)
        x_a = (attn_a @ v_f).transpose(1, 2).reshape(B_a, N_a, C_a)
        x_a = self.audio_proj(x_a)
        x_a = self.audio_proj_drop(x_a)

        attn_f = (q_f @ k_a.transpose(-2, -1)) * self.scale
        attn_f = attn_f.softmax(dim=-1)
        attn_f = self.frame_attn_drop(attn_f)
        x_f = (attn_f @ v_a).transpose(1, 2).reshape(B_f, N_f, C_f)
        x_f = self.frame_proj(x_f)
        x_f = self.frame_proj_drop(x_f)

        return x_f, x_a


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

    def forward(self, x, H, W, pos):
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
