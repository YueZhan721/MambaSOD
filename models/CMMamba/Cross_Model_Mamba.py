import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from CMMamba.mamba_simple import Mamba


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, ms, pan):
        b, c, h, w = ms.shape

        kv = self.kv_dwconv(self.kv(pan))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(ms))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm_cro1= LayerNorm(dim, LayerNorm_type)
        self.norm_cro2 = LayerNorm(dim, LayerNorm_type)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.cro = CrossAttention(dim,num_heads,bias)
        self.proj = nn.Conv2d(dim,dim,1,1,0)

    def forward(self, ms,pan):
        ms = ms+self.cro(self.norm_cro1(ms),self.norm_cro2(pan))
        ms = ms + self.ffn(self.norm2(ms))
        return ms


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


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
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
# ---------------------------------------------------------------------------------------------------------------------


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)


class PatchUnEmbed(nn.Module):
    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter

    def forward(self, x,x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, stride=4, in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):
        # (b,c,h,w)->(b,c*s*p,h//s,w//s)
        # (b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x


class CrossMamba(nn.Module):
    def __init__(self, dim, bimamba_type="v3"):
        super(CrossMamba, self).__init__()
        self.cross_mamba = Mamba(dim, bimamba_type=bimamba_type)
        self.norm1 = LayerNorm(dim, 'with_bias')
        self.norm2 = LayerNorm(dim, 'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depth-wise convolution

    def forward(self, ms, ms_resi, pan):  # ms (1,16384,32), pan (1,16384,32), ms_resi=0
        ms_resi = ms+ms_resi  # ms_resi (1,16384,32)  # todo ms_resi: come from last level

        ms = self.norm1(ms_resi)  # ms (1,16384,32)
        pan = self.norm2(pan)  # pan (1,16384,32)

        global_f = self.cross_mamba(self.norm1(ms), extra_emb=self.norm2(pan))  # global_f (1,16384,32)

        B, HW, C = global_f.shape  # B:1, HW:16384,C:32)
        H = W = int(math.sqrt(HW))
        # ms = global_f.transpose(1, 2).view(B, C, 128*8, 128*8)
        ms = global_f.transpose(1, 2).view(B, C, H, W)  # Reshape: (1,16384,32)-->(1, 32, 128,128)
        ms = self.dwconv(ms).flatten(2).transpose(1, 2)  # (1, 128,128, 32)-->(1,16384,32)
        return ms, ms_resi


class CrossMamba_(nn.Module):
    def __init__(self, dim):
        super(CrossMamba_, self).__init__()
        self.cross_mamba = Mamba(dim, bimamba_type="v3")
        self.norm1 = LayerNorm(dim, 'with_bias')
        self.norm2 = LayerNorm(dim, 'with_bias')
        self.norm3 = LayerNorm(dim, 'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depth-wise convolution

        self.reduce = nn.Conv2d(2*dim, dim, kernel_size=1)

    def forward(self, ms, pan):  # ms (10,7744,96), pan (10,7744,96)

        batch_size = ms.size(0)  # 获取批次大小，应该为10
        num_patches = ms.size(1)  # 获取patch数量，应该为7744
        channels = ms.size(2)  # 获取通道数，应该为96
        height_width = int(num_patches ** 0.5)  # 假设为88
        ms = ms.permute(0, 2, 1)  # 变为 (batch_size, channels, num_patches)
        ms = ms.view(batch_size, channels, height_width, height_width)
        pan = pan.permute(0, 2, 1)  # 同样处理
        pan = pan.view(batch_size, channels, height_width, height_width)
        concat = torch.cat((ms, pan), dim=1)  # 现在形状为 (batch_size, 2*channels, height, width)
        concat = self.reduce(concat)  # 输出形状为 (batch_size, channels, height, width)

        concat = concat.view(batch_size, channels, -1)  # 变为 (batch_size, channels, num_patches)
        concat = concat.permute(0, 2, 1)  # 变为 (batch_size, num_patches, channels)
        ms = ms.view(batch_size, channels, -1)
        ms = ms.permute(0, 2, 1)
        pan = pan.view(batch_size, channels, -1)
        pan = pan.permute(0, 2, 1)


        global_f = self.cross_mamba(self.norm1(ms), extra_emb=self.norm2(pan), concated=self.norm3(concat))  # global_f (1,16384,32)
        global_f = global_f + ms  # shortcut connection

        B, HW, C = global_f.shape  # B:1, HW:16384,C:32)
        H = W = int(math.sqrt(HW))
        ms = global_f.transpose(1, 2).view(B, C, H, W)  # Reshape: (1,16384,32)-->(1, 32, 128,128)
        ms = self.dwconv(ms)

        # ms = self.dwconv(ms).flatten(2).transpose(1, 2)  # (1, 128,128, 32)-->(1,16384,32)
        return ms


class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x+resi


class Net(nn.Module):
    def __init__(self, num_channels=None, base_filter=None, args=None):
        super(Net, self).__init__()
        base_filter = 32
        self.base_filter = base_filter
        self.stride = 1
        self.patch_size = 1

        self.pan_encoder = nn.Sequential(
            nn.Conv2d(1, base_filter, 3, 1, 1),
            HinResBlock(base_filter, base_filter),
            HinResBlock(base_filter, base_filter),
            HinResBlock(base_filter, base_filter))
        self.ms_encoder = nn.Sequential(
            nn.Conv2d(4, base_filter, 3, 1, 1),
            HinResBlock(base_filter, base_filter),
            HinResBlock(base_filter, base_filter),
            HinResBlock(base_filter, base_filter))

        self.embed_dim = base_filter*self.stride*self.patch_size
        self.ms_to_token = PatchEmbed(in_chans=base_filter, embed_dim=self.embed_dim, patch_size=self.patch_size, stride=self.stride)
        self.pan_to_token = PatchEmbed(in_chans=base_filter, embed_dim=self.embed_dim, patch_size=self.patch_size, stride=self.stride)
        self.deep_fusion1 = CrossMamba(self.embed_dim)  # todo 跨模态融合模块
        self.deep_fusion2 = CrossMamba(self.embed_dim)
        self.deep_fusion3 = CrossMamba(self.embed_dim)
        self.deep_fusion4 = CrossMamba(self.embed_dim)
        self.deep_fusion5 = CrossMamba(self.embed_dim)

        self.patchunembe = PatchUnEmbed(base_filter)

    def forward(self, ms, _, pan):  # ms(1,4,32,32), pan(1,1,128,128)

        ms_bic = F.interpolate(ms, scale_factor=4)  # ms(1,4,32,32)--> (1,4,128,128)
        ms_f = self.ms_encoder(ms_bic)  # (1,4,128,128)-->(1,32,128,128)

        b, c, h, w = ms_f.shape
        pan_f = self.pan_encoder(pan)  # (1,1,128,128)-->(1,32,128,128)

        ms_f = self.ms_to_token(ms_f)  # ms_f (1,32,128, 128)--> (1,16384,32)
        pan_f = self.pan_to_token(pan_f)  # pan_f (1,32,128, 128)--> (1,16384,32)
        residual_ms_f = 0  # todo 跨模态融合模块
        ms_f, residual_ms_f = self.deep_fusion1(ms_f, residual_ms_f, pan_f)  # ms_f(1,16384,32), pan(1,16384,32)->ms_f(1,16384,32)
        ms_f, residual_ms_f = self.deep_fusion2(ms_f, residual_ms_f, pan_f)  # ms_f(1,16384,32), pan(1,16384,32)->ms_f(1,16384,32)
        ms_f, residual_ms_f = self.deep_fusion3(ms_f, residual_ms_f, pan_f)
        ms_f, residual_ms_f = self.deep_fusion4(ms_f, residual_ms_f, pan_f)
        ms_f, residual_ms_f = self.deep_fusion5(ms_f, residual_ms_f, pan_f)
        ms_f = self.patchunembe(ms_f, (h, w))  # ms_f(1,16384,32)--> (1, 32, 128,128)
        return ms_f  # (1, 4, 128,128)


if __name__ == '__main__':
    # cuda or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 实例化模型得到分类结果
    lms_image = torch.randn(2, 4, 32, 32).to(device)
    bms_image = torch.randn(2, 4, 32, 32).to(device)
    pan_image = torch.randn(2, 1, 128, 128).to(device)

    model = Net().to(device)
    outputs = model(lms_image, bms_image, pan_image)
    print(outputs.shape)
