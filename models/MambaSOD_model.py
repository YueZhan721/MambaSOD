import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_
import math

from models.CMMamba.Cross_Model_Mamba import CrossMamba_
from models.CMMamba.Cross_Model_Mamba import PatchEmbed
from models.VMamba.classification.models import build_vssm_model
from models.VMamba.classification.config import get_config
from options import opt
import os

TRAIN_SIZE = 352

class PDFNet(nn.Module):
    def __init__(self, channel=32):
        super(PDFNet, self).__init__()

        # todo 1 Backbone model
        self.vmamba_config = get_config(opt)
        self.encoder_rgb = build_vssm_model(self.vmamba_config)
        self.encoder_depth = build_vssm_model(self.vmamba_config)
        if opt.pre:
              if os.path.isfile(opt.pre):
                print("=> loading checkpoint '{}'".format(opt.pre))
                checkpoint = torch.load(opt.pre)
                self.encoder_rgb.load_state_dict(checkpoint['model'], strict=False)
                self.encoder_depth.load_state_dict(checkpoint['model'], strict=False)
                print("=> loaded checkpoint")

        # todo 2 Cross-Modal Mamba module
        self.rawD_to_token1 = PatchEmbed(in_chans=96, embed_dim=96, patch_size=1, stride=1)
        self.rgb_to_token1 = PatchEmbed(in_chans=96, embed_dim=96, patch_size=1, stride=1)
        self.deep_fusion1 = CrossMamba_(96)  # todo 跨模态融合模块

        self.rawD_to_token2 = PatchEmbed(in_chans=96, embed_dim=96, patch_size=1, stride=1)
        self.rgb_to_token2 = PatchEmbed(in_chans=96, embed_dim=96, patch_size=1, stride=1)
        self.deep_fusion2 = CrossMamba_(96)  # todo 跨模态融合模块

        self.rawD_to_token3 = PatchEmbed(in_chans=192, embed_dim=192, patch_size=1, stride=1)
        self.rgb_to_token3 = PatchEmbed(in_chans=192, embed_dim=192, patch_size=1, stride=1)
        self.deep_fusion3 = CrossMamba_(192)  # todo 跨模态融合模块

        self.rawD_to_token4 = PatchEmbed(in_chans=384, embed_dim=384, patch_size=1, stride=1)
        self.rgb_to_token4 = PatchEmbed(in_chans=384, embed_dim=384, patch_size=1, stride=1)
        self.deep_fusion4 = CrossMamba_(384)  # todo 跨模态融合模块

        self.mcm4 = MCM(inc=768, outc=384)
        self.mcm3 = MCM(inc=384, outc=192)
        self.mcm2 = MCM(inc=192, outc=96)
        self.mcm1 = MCM_1(inc=96, outc=96)

        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(768),
            nn.GELU(),
            nn.Conv2d(in_channels=768, out_channels=1, kernel_size=1, stride=1)
        )

    def forward(self, x_rgb, x_depth):
        x_depth = x_depth.repeat(1, 3, 1, 1)
        x1_rgb, x2_rgb, x3_rgb, x4_rgb, x5_rgb = self.encoder_rgb(x_rgb)
        x1_depth, x2_depth, x3_depth, x4_depth, x5_depth = self.encoder_depth(x_depth)

        # layer1 merge
        x1_depth_token = self.rawD_to_token1(x1_depth)
        x1_rgb_token = self.rgb_to_token1(x1_rgb)
        x1_fusion = self.deep_fusion1(x1_rgb_token, x1_depth_token)  # (b,96,88,88)
        # layer1 merge end

        # layer2 merge
        x2_depth_token = self.rawD_to_token2(x2_depth)
        x2_rgb_token = self.rgb_to_token2(x2_rgb)
        x2_fusion = self.deep_fusion2(x2_rgb_token, x2_depth_token)  # (b,96,88,88)
        # layer2 merge end

        # layer3 merge
        x3_depth_token = self.rawD_to_token3(x3_depth)
        x3_rgb_token = self.rgb_to_token3(x3_rgb)
        x3_fusion = self.deep_fusion3(x3_rgb_token, x3_depth_token)  # (b,192,44,44)
        # layer3 merge end

        # layer4 merge
        x4_depth_token = self.rawD_to_token4(x4_depth)
        x4_rgb_token = self.rgb_to_token4(x4_rgb)
        x4_fusion = self.deep_fusion4(x4_rgb_token, x4_depth_token)  # (b,384,22,22)
        # layer4 merge end

        # layer5 merge
        x5_fusion = x5_rgb+x5_depth # (b,768,11,11)
        # layer5 merge end

        pred_5 = F.interpolate(self.predtrans(x5_fusion), TRAIN_SIZE, mode="bilinear", align_corners=True)
        pred_4, xf_4 = self.mcm4(x4_fusion, x5_fusion)
        pred_3, xf_3 = self.mcm3(x3_fusion, xf_4)
        pred_2, xf_2 = self.mcm2(x2_fusion, xf_3)
        pred_1, xf_1 = self.mcm1(x1_fusion, xf_2)

        return pred_1, pred_2, pred_3, pred_4, pred_5


    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

# Decoder
class MCM(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.rc = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )
        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1),
            nn.BatchNorm2d(outc),
            nn.GELU(),
            nn.Conv2d(in_channels=outc, out_channels=1, kernel_size=1)
        )

        self.rc2 = nn.Sequential(
            nn.Conv2d(in_channels=outc * 2, out_channels=outc, kernel_size=3, padding=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

        self.apply(self._init_weights)

    def forward(self, x1, x2):
        x2_upsample = self.upsample2(x2)  

        x2_rc = self.rc(x2_upsample)  
        shortcut = x2_rc

        x_cat = torch.cat((x1, x2_rc), dim=1)  
        x_mul = torch.mul(x1, x2_rc)
        x_forward = self.rc2(x_cat)  

        x_forward = x_forward + shortcut + x_mul
        pred = F.interpolate(self.predtrans(x_forward), TRAIN_SIZE, mode="bilinear", align_corners=True)  # 预测图

        return pred, x_forward

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

class MCM_1(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.rc = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )
        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1),
            nn.BatchNorm2d(outc),
            nn.GELU(),
            nn.Conv2d(in_channels=outc, out_channels=1, kernel_size=1)
        )

        self.rc2 = nn.Sequential(
            nn.Conv2d(in_channels=outc * 2, out_channels=outc, kernel_size=3, padding=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

        self.apply(self._init_weights)

    def forward(self, x1, x2):
        x2_rc = self.rc(x2)  
        shortcut = x2_rc

        x_cat = torch.cat((x1, x2_rc), dim=1)  
        x_mul = torch.mul(x1, x2_rc)
        x_forward = self.rc2(x_cat)  

        x_forward = x_forward + shortcut + x_mul
        pred = F.interpolate(self.predtrans(x_forward), TRAIN_SIZE, mode="bilinear", align_corners=True)  # 预测图

        return pred, x_forward

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