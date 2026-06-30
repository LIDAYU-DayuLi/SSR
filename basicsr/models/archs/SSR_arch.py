import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange, repeat
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from torchvision.transforms.functional import resize, to_pil_image
import numpy as np


def calc_init_centroid(images, num_spixels_w, num_spixels_h):
    B, C, H, W = images.shape
    device = images.device
    centroids = F.adaptive_avg_pool2d(images, (num_spixels_h, num_spixels_w))
    with torch.no_grad():
        labels = torch.arange(num_spixels_h * num_spixels_w, device=device).reshape(1, 1, num_spixels_h, num_spixels_w)
        init_label_map = F.interpolate(labels.float(), size=(H, W), mode="nearest").long()
        init_label_map = init_label_map.repeat(B, 1, 1, 1)
    centroids = centroids.reshape(B, C, -1)
    init_label_map = init_label_map.reshape(B, -1)
    return centroids, init_label_map


@torch.no_grad()
def get_abs_indices(init_label_map, num_spixels_w):
    b, n_pixel = init_label_map.shape
    device = init_label_map.device
    r = torch.arange(-1, 2.0, device=device)
    relative_idx = torch.cat([r - num_spixels_w, r, r + num_spixels_w], 0)
    abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1)
    abs_spix_indices = (init_label_map[:, None] + relative_idx[None, :, None]).reshape(-1)
    abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1)
    return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)


def ssn_iter(pixel_features, stoken_size=[16, 16], n_iter=2):
    B, C, H, W = pixel_features.shape
    sh, sw = stoken_size
    num_spixels_h = H // sh
    num_spixels_w = W // sw
    if num_spixels_h <= 0 or num_spixels_w <= 0:
        num_spixels_h = max(1, num_spixels_h)
        num_spixels_w = max(1, num_spixels_w)
    K = num_spixels_h * num_spixels_w
    spixel_features, init_label_map = calc_init_centroid(pixel_features, num_spixels_w, num_spixels_h)
    pixel_flat = pixel_features.reshape(B, C, -1)
    pixel_sq = (pixel_flat * pixel_flat).sum(dim=1, keepdim=True)

    for it in range(n_iter):
        cross = torch.bmm(spixel_features.permute(0, 2, 1), pixel_flat)
        sp_sq = (spixel_features * spixel_features).sum(dim=1, keepdim=True).permute(0, 2, 1)
        dist = sp_sq + pixel_sq - 2.0 * cross
        dist = torch.clamp(dist, min=0.0)
        affinity = (-dist).softmax(dim=1)

        if it < n_iter - 1:
            perm_pf = pixel_flat.permute(0, 2, 1).contiguous()
            new_sp = torch.bmm(affinity, perm_pf)
            denom = affinity.sum(dim=2, keepdim=True)
            spixel_features = (new_sp / (denom + 1e-16)).permute(0, 2, 1).contiguous()

    return affinity, K


class GenSP(nn.Module):
    def __init__(self, n_iter=2):
        super().__init__()
        self.n_iter = n_iter

    def forward(self, x, stoken_size):
        return ssn_iter(x, stoken_size, self.n_iter)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=8,
        d_conv=3,
        expand=2.,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.act = nn.GELU()

        self.x_proj = (
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.x_conv = nn.Conv1d(
            in_channels=(self.dt_rank + self.d_state * 2),
            out_channels=(self.dt_rank + self.d_state * 2),
            kernel_size=7,
            padding=3,
            groups=(self.dt_rank + self.d_state * 2),
        )

        self.dt_projs = (
            self.dt_init(
                self.dt_rank, self.d_inner,
                dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                **factory_kwargs
            ),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner)
        self.Ds = self.D_init(self.d_inner)
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.gen_sp = GenSP(n_iter=2)

        self.region_gate = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner),
            nn.Sigmoid()
        )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **kwargs):
        proj = nn.Linear(dt_rank, d_inner, bias=True, **kwargs)
        std = dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(proj.weight, -std, std)

        dt = torch.exp(
            torch.rand(d_inner, **kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        proj.bias.data.copy_(dt + torch.log(-torch.expm1(-dt)))
        return proj

    @staticmethod
    def A_log_init(d_state, d_inner):
        A = torch.arange(1, d_state + 1).float()
        A = A.unsqueeze(0).repeat(d_inner, 1)
        return nn.Parameter(torch.log(A))

    @staticmethod
    def D_init(d_inner):
        return nn.Parameter(torch.ones(d_inner))

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1

        x = x.view(B, 1, C, L)

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l",
            x,
            self.x_proj_weight
        )
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)

        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )

        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts,
            self.dt_projs_weight
        )

        xs = x.float().view(B, -1, L)
        dts = dts.float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        return y.view(B, -1, L)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W

        xz = self.in_proj(x.permute(0, 2, 3, 1))
        x_mid, z = xz.chunk(2, dim=-1)
        x_mid = x_mid.permute(0, 3, 1, 2).contiguous()
        x_mid = self.act(self.conv2d(x_mid))

        affinity, K = self.gen_sp(x_mid, stoken_size=[32, 32])
        sp_labels = affinity.argmax(dim=1)

        sort_idx = torch.argsort(sp_labels, dim=1)
        inv_idx = torch.argsort(sort_idx, dim=1)

        x_seq = x_mid.view(B, self.d_inner, L)
        x_seq_sorted = torch.gather(
            x_seq, 2, sort_idx.unsqueeze(1).expand(-1, self.d_inner, -1)
        )

        y_sorted = self.forward_core(x_seq_sorted.view(B, self.d_inner, H, W))

        y = torch.gather(
            y_sorted, 2, inv_idx.unsqueeze(1).expand(-1, self.d_inner, -1)
        ).view(B, self.d_inner, H, W)

        x_flat = x_mid.view(B, self.d_inner, L)
        sp_onehot = F.one_hot(sp_labels, num_classes=K).float()
        sp_onehot = sp_onehot.permute(0, 2, 1)

        region_feat = torch.bmm(sp_onehot, x_flat.transpose(1, 2))
        region_feat = region_feat / (sp_onehot.sum(-1, keepdim=True) + 1e-6)

        region_gate = self.region_gate(region_feat)
        region_gate = region_gate.permute(0, 2, 1)

        gate_pixel = torch.gather(
            region_gate, 2, sp_labels.unsqueeze(1).expand(-1, self.d_inner, -1)
        ).view(B, self.d_inner, H, W)

        y = y * gate_pixel

        y = y.permute(0, 2, 3, 1)
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y).permute(0, 3, 1, 2)

        return out


class S3M(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=False, LayerNorm_type='WithBias', att=False, idx=3, patch=128):
        super(S3M, self).__init__()
        self.att = att
        self.idx = idx
        if self.att:
            self.norm1 = LayerNorm(dim)
            self.attn = SS2D(d_model=dim, patch=patch)
        self.norm2 = LayerNorm(dim)
        self.ffn = FFN(dim, ffn_expansion_factor, bias)
        self.kernel_size = (patch, patch)

    def forward(self, x):
        if self.idx % 2 == 1:
            x = torch.flip(x, dims=(-2, -1)).contiguous()
        if self.idx % 2 == 0:
            x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()
        if self.att:
            x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


class SSR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[6, 6, 12],
                 ffn_expansion_factor=3,
                 bias=False):
        super(SSR, self).__init__()

        self.encoder = True
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = S3M(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias, att=True, idx=i,
                        patch=384)
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = S3M(dim=dim * 2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, att=True, idx=i,
                        patch=192)
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = S3M(dim=dim * 4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, att=True, idx=i,
                        patch=96)
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = S3M(dim=dim * 4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, att=True, idx=i,
                        patch=96)
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = S3M(dim=dim * 2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, att=True, idx=i,
                        patch=192)
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = S3M(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias, att=True, idx=i,
                        patch=384)
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, return_degrade=False):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = inp_dec_level2 + out_enc_level2
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = inp_dec_level1 + out_enc_level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        restored_img = self.output(out_dec_level1) + inp_img

        return restored_img