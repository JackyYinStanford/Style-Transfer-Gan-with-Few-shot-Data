import torch
import pdb
import torch.nn as nn
from .quant_base_layer import Eltwise
import functools
import torch.nn.functional as F


# from models.pix2pixnod_attention_model_for_tnn import Pix2PixNodAttentionModelTNN


class SEModule(nn.Module):
    def __init__(self, oup, reduction_ratio=4, use_fc=0):
        super(SEModule, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.use_fc = use_fc
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if use_fc:
            self.excitation = nn.Sequential(
                nn.Linear(oup, oup // self.reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Linear(oup // self.reduction_ratio, oup),
                nn.Sigmoid()
            )
        else:
            self.excitation = nn.Sequential(
                nn.Conv2d(oup, oup // self.reduction_ratio, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(oup // self.reduction_ratio, oup, kernel_size=1),
                nn.Sigmoid()
            )
        # self.relu = nn.ReLU()

    def forward(self, x):
        n, c, _, _ = x.shape
        out = self.avg_pool(x)
        if self.use_fc:
            out = self.excitation(out.view(n, c)).view(n, c, 1, 1)
        else:
            out = self.excitation(out)
        # out = self.relu(out + 3)/6.0
        res = x * out
        return res


###################################################################
### CBAM Module
###################################################################
# TODO bn --> norm_layer...
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


############################################################################################
class UpResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio=2, bn=True, norm_layer=nn.BatchNorm2d,
                 out_h=0, out_w=0, reduction_ratio=0, use_fc=0, use_v1=False, use_cbam=0):
        super(UpResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        use_bias = True

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        self.reduction_ratio = reduction_ratio
        self.use_fc = use_fc

        model = [
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=use_bias),
            norm_layer(hidden_dim),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=use_bias),
            norm_layer(hidden_dim),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=use_bias)
        ]
        if bn:
            model = model + [norm_layer(oup)]

        self.conv = nn.Sequential(*model)
        self.add = Eltwise()

        if out_h == 0 and out_w == 0:
            self.use_upsample = True
        else:
            self.use_upsample = False
            # assert out_h>0 and out_w>0
            self.upsample = nn.Upsample(size=(out_h, out_w), mode='bilinear')

    def forward(self, x, p=False):
        if self.use_upsample:
            out = F.upsample(x, scale_factor=2, mode='bilinear')
        else:
            out = self.upsample(x)

        if self.use_res_connect:
            res = self.add(out, self.conv(out))
        else:
            res = self.conv(out)

        return res


class DownResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio=1, bn=True, norm_layer=nn.BatchNorm2d,
                 reduction_ratio=0, use_fc=0, use_v1=0, use_cbam=0, first_conv='large'):
        super(DownResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        # 用IN的时候采用bias
        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d

        use_bias = True

        if inp == 3 and expand_ratio == 1:
            expand_ratio = 2

        hidden_dim = round(inp * expand_ratio)
        #         if inp == 3:
        #             if hidden_dim >= 16:
        #                 hidden_dim = 16
        #             else:
        #                 hidden_dim = 8
        if inp == 3:
            if first_conv == 'large':
                hidden_dim = 16
            else:
                hidden_dim = 8

        self.use_res_connect = self.stride == 1 and inp == oup  # 只有步长为1和输入输出channel相等的时候才用resBlock

        self.reduction_ratio = reduction_ratio
        self.use_fc = use_fc

        model = [nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=use_bias),
                 norm_layer(hidden_dim),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=use_bias),
                 norm_layer(hidden_dim),
                 nn.ReLU(inplace=True),
                 # pw-linear
                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=use_bias)]
        if bn:
            model = model + [norm_layer(oup)]

        self.conv = nn.Sequential(*model)

    def forward(self, x, p=False):
        if self.use_res_connect:
            res = x + self.conv(x)
        else:
            res = self.conv(x)

        return res


class MobileUnetV2(nn.Module):
    def __init__(self, layer_size=6, channel=8, expand=2, gpu=1, norm_layer=nn.BatchNorm2d, use_residual=True,
                 with_mask=False, input_h=0, input_w=0, reduction_ratio=0, use_fc=0, use_cbam=0, with_warp=False):
        super(MobileUnetV2, self).__init__()

        # assert input_h != 0 and input_w != 0
        self.input_h = input_h
        self.input_w = input_w

        self.layer_size = layer_size
        self.use_residual = use_residual
        self.with_warp = with_warp
        self.with_mask = with_mask

        # assert not(reduction_ratio > 0 and use_cbam > 0)

        # channel = 4 [1.5m]
        self.enc_0 = DownResidual(3, 64 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_1 = nn.ReLU(inplace=True)
        self.enc_1 = DownResidual(64 // channel, 128 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_2 = nn.ReLU(inplace=True)
        self.enc_2 = DownResidual(128 // channel, 256 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_3 = nn.ReLU(inplace=True)
        self.enc_3 = DownResidual(256 // channel, 512 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_4 = nn.ReLU(inplace=True)
        self.enc_4 = DownResidual(512 // channel, 512 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_5 = nn.ReLU(inplace=True)
        self.enc_5 = DownResidual(512 // channel, 512 // channel, 2, expand_ratio=expand, bn=False,
                                  norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_relu_5 = nn.ReLU(inplace=True)
        # ResidualBlock
        self.dec_5 = UpResidual(512 // channel, 512 // channel, 1, expand_ratio=expand,
                                norm_layer=norm_layer,
                                out_h=self.input_h // 32, out_w=self.input_w // 32,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        # ResidualBlock
        self.dec_add_4 = Eltwise()
        self.dec_relu_4 = nn.ReLU(inplace=True)
        self.dec_4 = UpResidual(512 // channel * 1, 512 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 16, out_w=self.input_w // 16,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_3 = Eltwise()
        self.dec_relu_3 = nn.ReLU(inplace=True)
        self.dec_3 = UpResidual(512 // channel * 1, 256 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 8, out_w=self.input_w // 8,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_2 = Eltwise()
        self.dec_relu_2 = nn.ReLU(inplace=True)
        self.dec_2 = UpResidual(256 // channel * 1, 128 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 4, out_w=self.input_w // 4,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_1 = Eltwise()
        self.dec_relu_1 = nn.ReLU(inplace=True)
        self.dec_1 = UpResidual(128 // channel * 1, 64 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 2, out_w=self.input_w // 2,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_1_hair = UpResidual(128 // channel * 1, 64 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                     out_h=self.input_h // 2, out_w=self.input_w // 2,
                                     reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_0 = Eltwise()
        self.dec_relu_0 = nn.ReLU(inplace=True)
        self.dec_add_0_hair = Eltwise()
        self.dec_relu_0_hair = nn.ReLU(inplace=True)

        if with_hair:
            self.dec_0 = UpResidual(64 // channel * 1, 4, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)

            self.dec_0_hair = UpResidual(64 // channel * 1, 4, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                         out_h=self.input_h, out_w=self.input_w,
                                         reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False,
                                         use_cbam=use_cbam)

        elif with_warp:
            self.dec_0 = UpResidual(64 // channel * 1, 6, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)

        elif with_mask:
            self.dec_0 = UpResidual(64 // channel * 1, 4, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)
        else:
            self.dec_0 = UpResidual(64 // channel * 1, 3, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)

        self.tanh = nn.Tanh()
        self.dec_add_image = Eltwise()

    def forward(self, image, mask=None):
        h_dict = {}

        h = self.enc_0(image, p=True)
        h_dict['cat_0'] = h

        h = self.enc_relu_1(h)
        h = self.enc_1(h)
        # h = self.enc_bn_1(h)
        h_dict['cat_1'] = h

        h = self.enc_relu_2(h)
        h = self.enc_2(h)
        # h = self.enc_bn_2(h)
        h_dict['cat_2'] = h

        h = self.enc_relu_3(h)
        h = self.enc_3(h)
        # h = self.enc_bn_3(h)
        h_dict['cat_3'] = h

        h = self.enc_relu_4(h)
        h = self.enc_4(h)
        # h = self.enc_bn_4(h)
        h_dict['cat_4'] = h

        h = self.enc_relu_5(h)
        h = self.enc_5(h)

        h = self.dec_relu_5(h)
        h = self.dec_5(h, p=True)
        # h = self.dec_bn_5(h)
        h = self.dec_relu_4(h)
        h = self.dec_add_4(h, h_dict['cat_4'])
        # h = torch.cat([h, h_dict['cat_4']], 1)

        h = self.dec_4(h)
        # h = self.dec_bn_4(h)
        h = self.dec_relu_3(h)
        h = self.dec_add_3(h, h_dict['cat_3'])
        # h = torch.cat([h, h_dict['cat_3']], 1)

        h = self.dec_3(h)
        # h = self.dec_bn_3(h)
        h = self.dec_relu_2(h)
        h = self.dec_add_2(h, h_dict['cat_2'])
        # h = torch.cat([h, h_dict['cat_2']], 1)

        h = self.dec_2(h)
        # h = self.dec_bn_2(h)
        h = self.dec_relu_1(h)
        h = self.dec_add_1(h, h_dict['cat_1'])
        # h = torch.cat([h, h_dict['cat_1']], 1)

        h_face = self.dec_1(h)
        # h = self.dec_bn_1(h)
        h_face = self.dec_relu_0(h_face)
        h_face = self.dec_add_0(h_face, h_dict['cat_0'])

        h_hair = self.dec_1_hair(h)
        h_hair = self.dec_relu_0_hair(h_hair)
        h_hair = self.dec_add_0_hair(h_hair, h_dict['cat_0'])
        # h = torch.cat([h, h_dict['cat_0']], 1)

        h_face = self.dec_0(h_face)
        h_hair = self.dec_0_hair(h_hair)

        if self.with_hair:
            h_face = self.tanh(h_face)
            hair_rgba = self.tanh(h_hair)

            face_rgb, mask = h_face[:, :3, :, :], h_face[:, 3:, :, :]

            self.rgb = face_rgb
            self.alpha = mask
            self.hair_rgba = hair_rgba

            return face_rgb, mask, hair_rgba

        if self.with_warp:
            h = self.tanh(h_face)
            h, mask, flow = h[:, :3, :, :], h[:, 3:4, :, :], h[:, 4:, :, :]
            # mask = (mask + 1.0)/2.0
            self.rgb = h
            # h = h * mask + image * (1.-mask)
            self.alpha = mask
            self.flow = flow
            return h, mask, flow

        if self.with_mask:
            h = self.tanh(h_face)
            h, mask = h[:, :3, :, :], h[:, 3:, :, :]
            # mask = (mask + 1.0)/2.0
            self.rgb = h
            # h = h * mask + image * (1.-mask)
            self.alpha = mask
            return h, mask
            # return h            

        if self.use_residual:
            h = self.dec_add_image(h, image)
            h = torch.clamp(h, min=-1.0, max=1.0)
        else:
            h = self.tanh(h)
        return h


class MobileUnetV3(nn.Module):
    def __init__(self, layer_size=6, channel=8, expand=2, gpu=1, norm_layer=nn.BatchNorm2d, use_residual=True,
                 with_mask=False, input_h=0, input_w=0, reduction_ratio=0, use_fc=0, use_cbam=0, with_warp=False,
                 with_hair=False):
        super(MobileUnetV3, self).__init__()

        # assert input_h != 0 and input_w != 0
        self.input_h = input_h
        self.input_w = input_w

        self.layer_size = layer_size
        self.use_residual = use_residual
        self.with_warp = with_warp
        self.with_hair = with_hair
        self.with_mask = with_mask

        # assert not(reduction_ratio > 0 and use_cbam > 0)

        # channel = 4 [1.5m]
        # channel = 32(ngf=16)
        self.enc_0 = DownResidual(3, 64 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_1 = nn.ReLU(inplace=True)
        self.enc_1 = DownResidual(64 // channel, 128 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_2 = nn.ReLU(inplace=True)
        self.enc_2 = DownResidual(128 // channel, 256 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_3 = nn.ReLU(inplace=True)
        self.enc_3 = DownResidual(256 // channel, 512 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_4 = nn.ReLU(inplace=True)
        self.enc_4 = DownResidual(512 // channel, 512 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_5 = nn.ReLU(inplace=True)
        self.enc_5 = DownResidual(512 // channel, 512 // channel, 2, expand_ratio=expand, bn=False,
                                  norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_relu_5 = nn.ReLU(inplace=True)
        self.dec_5 = UpResidual(512 // channel, 512 // channel, 1, expand_ratio=expand,  # res
                                norm_layer=norm_layer,
                                out_h=self.input_h // 32, out_w=self.input_w // 32,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_4 = Eltwise()
        self.dec_relu_4 = nn.ReLU(inplace=True)
        self.dec_4 = UpResidual(512 // channel * 1, 512 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                # res
                                out_h=self.input_h // 16, out_w=self.input_w // 16,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_3 = Eltwise()
        self.dec_relu_3 = nn.ReLU(inplace=True)
        self.dec_3 = UpResidual(512 // channel * 1, 256 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 8, out_w=self.input_w // 8,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_2 = Eltwise()
        self.dec_relu_2 = nn.ReLU(inplace=True)
        self.dec_2 = UpResidual(256 // channel * 1, 128 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 4, out_w=self.input_w // 4,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_1 = Eltwise()
        self.dec_relu_1 = nn.ReLU(inplace=True)

        self.dec_1 = UpResidual(128 // channel * 1, 64 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 2, out_w=self.input_w // 2,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_0 = Eltwise()
        self.dec_relu_0 = nn.ReLU(inplace=True)

        if with_hair:
            self.dec_0 = UpResidual(64 // channel * 1, 4, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)

            self.dec_0_hair = UpResidual(64 // channel * 1, 4, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                         out_h=self.input_h, out_w=self.input_w,
                                         reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False,
                                         use_cbam=use_cbam)

        elif with_warp:
            self.dec_0 = UpResidual(64 // channel * 1, 6, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)

        elif with_mask:
            self.dec_0 = UpResidual(64 // channel * 1, 4, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)
        else:
            self.dec_0 = UpResidual(64 // channel * 1, 3, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)

        self.tanh = nn.Tanh()
        self.dec_add_image = Eltwise()

    def forward(self, image, mask=None):
        h_dict = {}

        h = self.enc_0(image, p=True)
        h_dict['cat_0'] = h

        h = self.enc_relu_1(h)
        h = self.enc_1(h)
        h_dict['cat_1'] = h

        h = self.enc_relu_2(h)
        h = self.enc_2(h)
        h_dict['cat_2'] = h

        h = self.enc_relu_3(h)
        h = self.enc_3(h)
        h_dict['cat_3'] = h

        h = self.enc_relu_4(h)
        h = self.enc_4(h)
        h_dict['cat_4'] = h

        h = self.enc_relu_5(h)
        h = self.enc_5(h)

        h = self.dec_relu_5(h)
        h = self.dec_5(h, p=True)

        h = self.dec_relu_4(h)

        h = self.dec_add_4(h, h_dict['cat_4'])

        h = self.dec_4(h)
        h = self.dec_relu_3(h)
        h = self.dec_add_3(h, h_dict['cat_3'])

        h = self.dec_3(h)
        h = self.dec_relu_2(h)
        h = self.dec_add_2(h, h_dict['cat_2'])

        h_face = self.dec_2(h)
        h_face = self.dec_relu_1(h_face)
        h_face = self.dec_add_1(h_face, h_dict['cat_1'])

        h_face = self.dec_1(h_face)
        h_face = self.dec_relu_0(h_face)
        h_face = self.dec_add_0(h_face, h_dict['cat_0'])

        h_face = self.dec_0(h_face)

        if self.with_warp:
            h = self.tanh(h_face)
            h, mask, flow = h[:, :3, :, :], h[:, 3:4, :, :], h[:, 4:, :, :]
            # mask = (mask + 1.0)/2.0
            self.rgb = h
            # h = h * mask + image * (1.-mask)
            self.alpha = mask
            self.flow = flow
            return h, mask, flow

        if self.with_mask:
            h = self.tanh(h_face)
            h, mask = h[:, :3, :, :], h[:, 3:, :, :]
            # mask = (mask + 1.0)/2.0
            self.rgb = h
            # h = h * mask + image * (1.-mask)
            self.alpha = mask
            return h, mask
            # return h

        if self.use_residual:
            h = self.dec_add_image(h, image)
            h = torch.clamp(h, min=-1.0, max=1.0)
        else:
            h = self.tanh(h)
        return h

class MobileUnetV4(nn.Module):
    def __init__(self, layer_size=6, channel=8, expand=2, gpu=1, norm_layer=nn.BatchNorm2d, use_residual=True,
                 with_mask=False, input_h=0, input_w=0, reduction_ratio=0, use_fc=0, use_cbam=0, with_warp=False,
                 with_hair=False):
        super(MobileUnetV4, self).__init__()

        # assert input_h != 0 and input_w != 0
        self.input_h = input_h
        self.input_w = input_w

        self.layer_size = layer_size
        self.use_residual = use_residual
        self.with_warp = with_warp
        self.with_hair = with_hair
        self.with_mask = with_mask

        # assert not(reduction_ratio > 0 and use_cbam > 0)

        # channel = 4 [1.5m]
        # channel = 32(ngf=16)
        self.enc_0 = DownResidual(3, 64 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_1 = nn.ReLU(inplace=True)
        self.enc_1 = DownResidual(64 // channel, 128 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_2 = nn.ReLU(inplace=True)
        self.enc_2 = DownResidual(128 // channel, 256 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_3 = nn.ReLU(inplace=True)
        self.enc_3 = DownResidual(256 // channel, 512 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_4 = nn.ReLU(inplace=True)
        self.enc_4 = DownResidual(512 // channel, 512 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_5 = nn.ReLU(inplace=True)
        self.enc_5 = DownResidual(512 // channel, 512 // channel, 2, expand_ratio=expand, bn=False,
                                  norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_relu_5 = nn.ReLU(inplace=True)
        self.dec_5 = UpResidual(512 // channel, 512 // channel, 1, expand_ratio=expand,  # res
                                norm_layer=norm_layer,
                                out_h=self.input_h // 32, out_w=self.input_w // 32,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_4 = Eltwise()
        self.dec_relu_4 = nn.ReLU(inplace=True)
        self.dec_4 = UpResidual(512 // channel * 1, 512 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                # res
                                out_h=self.input_h // 16, out_w=self.input_w // 16,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_3 = Eltwise()
        self.dec_relu_3 = nn.ReLU(inplace=True)
        self.dec_3 = UpResidual(512 // channel * 1, 256 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 8, out_w=self.input_w // 8,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_2 = Eltwise()
        self.dec_relu_2 = nn.ReLU(inplace=True)
        self.dec_2 = UpResidual(256 // channel * 1, 128 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 4, out_w=self.input_w // 4,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_1 = Eltwise()
        self.dec_relu_1 = nn.ReLU(inplace=True)

        self.dec_1 = UpResidual(128 // channel * 1, 64 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 2, out_w=self.input_w // 2,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_0 = Eltwise()
        self.dec_relu_0 = nn.ReLU(inplace=True)

        if with_hair:
            self.dec_0 = UpResidual(64 // channel * 1, 4, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)

            self.dec_0_hair = UpResidual(64 // channel * 1, 4, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                         out_h=self.input_h, out_w=self.input_w,
                                         reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False,
                                         use_cbam=use_cbam)

        elif with_warp:
            self.dec_0 = UpResidual(64 // channel * 1, 6, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)

        elif with_mask:
            self.dec_0 = UpResidual(64 // channel * 1, 4, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)
        else:
            self.dec_0 = UpResidual(64 // channel * 1, 3, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)

        self.tanh = nn.Tanh()
        self.dec_add_image = Eltwise()

    def forward(self, image, mask=None):
        h_dict = {}

        h = self.enc_0(image, p=True)
        h_dict['cat_0'] = h

        h = self.enc_relu_1(h)
        h = self.enc_1(h)
        h_dict['cat_1'] = h

        h = self.enc_relu_2(h)
        h = self.enc_2(h)
        h_dict['cat_2'] = h

        h = self.enc_relu_3(h)
        h = self.enc_3(h)
        h_dict['cat_3'] = h

        h = self.enc_relu_4(h)
        h = self.enc_4(h)
        h_dict['cat_4'] = h

        h = self.enc_relu_5(h)
        h = self.enc_5(h)

        h = self.dec_relu_5(h)
        h = self.dec_5(h, p=True)

        h = self.dec_relu_4(h)

        h = self.dec_add_4(h, h_dict['cat_4'])

        h = self.dec_4(h)
        h = self.dec_relu_3(h)
        h = self.dec_add_3(h, h_dict['cat_3'])

        h = self.dec_3(h)
        h = self.dec_relu_2(h)
        h = self.dec_add_2(h, h_dict['cat_2'])

        h_face = self.dec_2(h)
        h_face = self.dec_relu_1(h_face)
        h_face = self.dec_add_1(h_face, h_dict['cat_1'])

        h_face = self.dec_1(h_face)
        h_face = self.dec_relu_0(h_face)
        h_face = self.dec_add_0(h_face, h_dict['cat_0'])

        h_face = self.dec_0(h_face)

        if self.with_warp:
            h = self.tanh(h_face)
            h, mask, flow = h[:, :3, :, :], h[:, 3:4, :, :], h[:, 4:, :, :]
            # mask = (mask + 1.0)/2.0
            self.rgb = h
            # h = h * mask + image * (1.-mask)
            self.alpha = mask
            self.flow = flow
            return h, mask, flow

        if self.with_mask:
            h = self.tanh(h_face)
            h, mask = h[:, :3, :, :], h[:, 3:, :, :]
            # mask = (mask + 1.0)/2.0
            self.rgb = h
            # h = h * mask + image * (1.-mask)
            self.alpha = mask
            return h, mask
            # return h

        if self.use_residual:
            h = self.dec_add_image(h, image)
            h = torch.clamp(h, min=-1.0, max=1.0)
        else:
            h = self.tanh(h)
        return h

class MobileUnetV3_SE(nn.Module):
    def __init__(self, layer_size=6, channel=8, expand=2, gpu=1, norm_layer=nn.BatchNorm2d, use_residual=True,
                 with_mask=False, input_h=0, input_w=0, reduction_ratio=0, use_fc=0, use_cbam=0, with_warp=False,
                 with_hair=False):
        super(MobileUnetV3_SE, self).__init__()

        # assert input_h != 0 and input_w != 0
        self.input_h = input_h
        self.input_w = input_w

        self.layer_size = layer_size
        self.use_residual = use_residual
        self.with_warp = with_warp
        self.with_hair = with_hair
        self.with_mask = with_mask

        # assert not(reduction_ratio > 0 and use_cbam > 0)

        # channel = 4 [1.5m]
        self.enc_0 = DownResidual(3, 64 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)
        self.se_layer_0 = SEModule(64 // channel, 4, use_fc=0)

        self.enc_relu_1 = nn.ReLU(inplace=True)
        self.enc_1 = DownResidual(64 // channel, 128 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)
        self.se_layer_1 = SEModule(128 // channel, 4, use_fc=0)

        self.enc_relu_2 = nn.ReLU(inplace=True)
        self.enc_2 = DownResidual(128 // channel, 256 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)
        self.se_layer_2 = SEModule(256 // channel, 4, use_fc=0)

        self.enc_relu_3 = nn.ReLU(inplace=True)
        self.enc_3 = DownResidual(256 // channel, 512 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_4 = nn.ReLU(inplace=True)
        self.enc_4 = DownResidual(512 // channel, 512 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_5 = nn.ReLU(inplace=True)
        self.enc_5 = DownResidual(512 // channel, 512 // channel, 2, expand_ratio=expand, bn=False,
                                  norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_relu_5 = nn.ReLU(inplace=True)
        self.dec_5 = UpResidual(512 // channel, 512 // channel, 1, expand_ratio=expand,
                                norm_layer=norm_layer,
                                out_h=self.input_h // 32, out_w=self.input_w // 32,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_4 = Eltwise()
        self.dec_relu_4 = nn.ReLU(inplace=True)
        self.dec_4 = UpResidual(512 // channel * 1, 512 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 16, out_w=self.input_w // 16,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_3 = Eltwise()
        self.dec_relu_3 = nn.ReLU(inplace=True)
        self.dec_3 = UpResidual(512 // channel * 1, 256 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 8, out_w=self.input_w // 8,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_2 = Eltwise()
        self.dec_relu_2 = nn.ReLU(inplace=True)
        self.dec_2 = UpResidual(256 // channel * 1, 128 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 4, out_w=self.input_w // 4,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_1 = Eltwise()
        self.dec_relu_1 = nn.ReLU(inplace=True)

        self.dec_1 = UpResidual(128 // channel * 1, 64 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 2, out_w=self.input_w // 2,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_add_0 = Eltwise()
        self.dec_relu_0 = nn.ReLU(inplace=True)

        if with_warp:
            self.dec_0 = UpResidual(64 // channel * 1, 6, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)

        elif with_mask:
            self.dec_0 = UpResidual(64 // channel * 1, 4, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)
        else:
            self.dec_0 = UpResidual(64 // channel * 1, 3, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)

        self.tanh = nn.Tanh()
        self.dec_add_image = Eltwise()

    def forward(self, image, mask=None):
        h_dict = {}

        h = self.enc_0(image, p=True)
        # h_dict['cat_0'] = h
        h_dict['cat_se_0'] = self.se_layer_0(h)

        h = self.enc_relu_1(h.clone())
        h = self.enc_1(h)
        # h_dict['cat_1'] = h
        h_dict['cat_se_1'] = self.se_layer_1(h)

        h = self.enc_relu_2(h.clone())
        h = self.enc_2(h)
        # h_dict['cat_2'] = h
        h_dict['cat_se_2'] = self.se_layer_2(h)

        h = self.enc_relu_3(h.clone())
        h = self.enc_3(h)
        h_dict['cat_3'] = h

        h = self.enc_relu_4(h)
        h = self.enc_4(h)
        h_dict['cat_4'] = h

        h = self.enc_relu_5(h)
        h = self.enc_5(h)

        h = self.dec_relu_5(h)
        h = self.dec_5(h, p=True)
        h = self.dec_relu_4(h)
        h = self.dec_add_4(h, h_dict['cat_4'])

        h = self.dec_4(h)
        h = self.dec_relu_3(h)
        h = self.dec_add_3(h, h_dict['cat_3'])

        h = self.dec_3(h)
        h = self.dec_relu_2(h)
        h_face = self.dec_add_2(h, h_dict['cat_se_2'])

        h_face = self.dec_2(h_face)
        h_face = self.dec_relu_1(h_face)
        h_face = self.dec_add_1(h_face, h_dict['cat_se_1'])

        h_hair = self.dec_2_hair(h_hair)
        h_hair = self.dec_relu_1_hair(h_hair)
        h_hair = self.dec_add_1_hair(h_hair, h_dict['cat_se_1_hair'])

        h_face = self.dec_1(h_face)
        h_face = self.dec_relu_0(h_face)
        h_face = self.dec_add_0(h_face, h_dict['cat_se_0'])

        if self.with_warp:
            h = self.tanh(h_face)
            h, mask, flow = h[:, :3, :, :], h[:, 3:4, :, :], h[:, 4:, :, :]
            # mask = (mask + 1.0)/2.0
            self.rgb = h
            # h = h * mask + image * (1.-mask)
            self.alpha = mask
            self.flow = flow
            return h, mask, flow

        if self.with_mask:
            h = self.tanh(h_face)
            h, mask = h[:, :3, :, :], h[:, 3:, :, :]
            # mask = (mask + 1.0)/2.0
            self.rgb = h
            # h = h * mask + image * (1.-mask)
            self.alpha = mask
            return h, mask
            # return h

        if self.use_residual:
            h = self.dec_add_image(h, image)
            h = torch.clamp(h, min=-1.0, max=1.0)
        else:
            h = self.tanh(h)
        return h


class MobileUnetTNN(nn.Module):
    def __init__(self, layer_size=6, channel=8, expand=2, gpu=1, norm_layer=nn.BatchNorm2d, use_residual=True,
                 with_warp=False, with_mask=False, input_h=0, input_w=0, reduction_ratio=0, use_fc=0, use_cbam=0):
        super(MobileUnetTNN, self).__init__()

        # assert input_h != 0 and input_w != 0
        self.input_h = input_h
        self.input_w = input_w

        self.layer_size = layer_size
        self.use_residual = use_residual

        # assert not(reduction_ratio > 0 and use_cbam > 0)

        # channel = 4 [1.5m]
        self.enc_0 = DownResidual(3, 64 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.enc_relu_1 = nn.ReLU(inplace=True)
        self.enc_1 = DownResidual(64 // channel, 128 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)
        # self.enc_bn_1 = nn.BatchNorm2d(128 // channel)

        self.enc_relu_2 = nn.ReLU(inplace=True)
        self.enc_2 = DownResidual(128 // channel, 256 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)
        # self.enc_bn_2 = nn.BatchNorm2d(256 // channel)

        self.enc_relu_3 = nn.ReLU(inplace=True)
        self.enc_3 = DownResidual(256 // channel, 512 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)
        # self.enc_bn_3 = nn.BatchNorm2d(512 // channel)

        self.enc_relu_4 = nn.ReLU(inplace=True)
        self.enc_4 = DownResidual(512 // channel, 512 // channel, 2, expand_ratio=expand, norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)
        # self.enc_bn_4 = nn.BatchNorm2d(512 // channel)

        self.enc_relu_5 = nn.ReLU(inplace=True)
        self.enc_5 = DownResidual(512 // channel, 512 // channel, 2, expand_ratio=expand, bn=False,
                                  norm_layer=norm_layer,
                                  reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)

        self.dec_relu_5 = nn.ReLU(inplace=True)
        self.dec_5 = UpResidual(512 // channel, 512 // channel, 1, expand_ratio=expand,
                                norm_layer=norm_layer,
                                out_h=self.input_h // 32, out_w=self.input_w // 32,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)
        # self.dec_bn_5 = nn.BatchNorm2d(512 // channel)

        self.dec_add_4 = Eltwise()
        self.dec_relu_4 = nn.ReLU(inplace=True)
        # self.dec_4 = UpResidual(512 // channel * 2, 512 // channel, 1, expand_ratio=expand)
        self.dec_4 = UpResidual(512 // channel * 1, 512 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 16, out_w=self.input_w // 16,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)
        # self.dec_bn_4 = nn.BatchNorm2d(512 // channel)

        self.dec_add_3 = Eltwise()
        self.dec_relu_3 = nn.ReLU(inplace=True)
        # self.dec_3 = UpResidual(512 // channel * 2, 256 // channel, 1, expand_ratio=expand)
        self.dec_3 = UpResidual(512 // channel * 1, 256 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 8, out_w=self.input_w // 8,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)
        # self.dec_bn_3 = nn.BatchNorm2d(256 // channel)

        self.dec_add_2 = Eltwise()
        self.dec_relu_2 = nn.ReLU(inplace=True)
        # self.dec_2 = UpResidual(256 // channel * 2, 128 // channel, 1, expand_ratio=expand)
        self.dec_2 = UpResidual(256 // channel * 1, 128 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 4, out_w=self.input_w // 4,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)
        # self.dec_bn_2 = nn.BatchNorm2d(128 // channel)

        self.dec_add_1 = Eltwise()
        self.dec_relu_1 = nn.ReLU(inplace=True)
        # self.dec_1 = UpResidual(128 // channel * 2, 64 // channel, 1, expand_ratio=expand)
        self.dec_1 = UpResidual(128 // channel * 1, 64 // channel, 1, expand_ratio=expand, norm_layer=norm_layer,
                                out_h=self.input_h // 2, out_w=self.input_w // 2,
                                reduction_ratio=reduction_ratio, use_fc=use_fc, use_cbam=use_cbam)
        # self.dec_bn_1 = nn.BatchNorm2d(64 // channel)

        self.dec_add_0 = Eltwise()
        self.dec_relu_0 = nn.ReLU(inplace=True)
        # self.dec_0 = UpResidual(64 // channel * 2, 3, 1, expand_ratio=expand, bn=False)
        self.with_mask = with_mask
        self.with_warp = with_warp

        if with_warp:
            self.dec_0 = UpResidual(64 // channel * 1, 6, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)
        elif with_mask:
            self.dec_0 = UpResidual(64 // channel * 1, 4, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)
        else:
            self.dec_0 = UpResidual(64 // channel * 1, 3, 1, expand_ratio=expand, bn=False, norm_layer=norm_layer,
                                    out_h=self.input_h, out_w=self.input_w,
                                    reduction_ratio=reduction_ratio, use_fc=use_fc, use_v1=False, use_cbam=use_cbam)
        self.tanh = nn.Tanh()
        self.dec_add_image = Eltwise()

    def forward(self, image, mask=None):
        h_dict = {}

        h = self.enc_0(image, p=True)
        h_dict['cat_0'] = h

        h = self.enc_relu_1(h)
        h = self.enc_1(h)
        # h = self.enc_bn_1(h)
        h_dict['cat_1'] = h

        h = self.enc_relu_2(h)
        h = self.enc_2(h)
        # h = self.enc_bn_2(h)
        h_dict['cat_2'] = h

        h = self.enc_relu_3(h)
        h = self.enc_3(h)
        # h = self.enc_bn_3(h)
        h_dict['cat_3'] = h

        h = self.enc_relu_4(h)
        h = self.enc_4(h)
        # h = self.enc_bn_4(h)
        h_dict['cat_4'] = h

        h = self.enc_relu_5(h)
        h = self.enc_5(h)

        h = self.dec_relu_5(h)
        h = self.dec_5(h, p=True)
        # h = self.dec_bn_5(h)
        h = self.dec_relu_4(h)
        h = self.dec_add_4(h, h_dict['cat_4'])
        # h = torch.cat([h, h_dict['cat_4']], 1)

        h = self.dec_4(h)
        # h = self.dec_bn_4(h)
        h = self.dec_relu_3(h)
        h = self.dec_add_3(h, h_dict['cat_3'])
        # h = torch.cat([h, h_dict['cat_3']], 1)

        h = self.dec_3(h)
        # h = self.dec_bn_3(h)
        h = self.dec_relu_2(h)
        h = self.dec_add_2(h, h_dict['cat_2'])
        # h = torch.cat([h, h_dict['cat_2']], 1)

        h = self.dec_2(h)
        # h = self.dec_bn_2(h)
        h = self.dec_relu_1(h)
        h = self.dec_add_1(h, h_dict['cat_1'])
        # h = torch.cat([h, h_dict['cat_1']], 1)

        h = self.dec_1(h)
        # h = self.dec_bn_1(h)
        h = self.dec_relu_0(h)
        h = self.dec_add_0(h, h_dict['cat_0'])
        # h = torch.cat([h, h_dict['cat_0']], 1)

        h = self.dec_0(h)

        if self.with_warp:
            h = self.tanh(h)
            h, mask, flow = h[:, :3, :, :], h[:, 3:4, :, :], h[:, 4:, :, :]
            # mask = (mask + 1.0)/2.0
            self.rgb = h
            # h = h * mask + image * (1.-mask)
            self.alpha = mask
            self.flow = flow
            return h, mask, flow

        elif self.with_mask:
            h = self.tanh(h)
            h, mask = h[:, :3, :, :], h[:, 3:, :, :]
            # mask = (mask + 1.0)/2.0
            self.rgb = h
            # h = h * mask + image * (1.-mask)
            self.alpha = mask
            return h, mask
            # return h

        if self.use_residual:
            h = self.dec_add_image(h, image)
            h = torch.clamp(h, min=-1.0, max=1.0)
        else:
            h = self.tanh(h)
        return h
