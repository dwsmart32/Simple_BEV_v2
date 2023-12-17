import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")

import utils.geom
import utils.vox
import utils.misc
import utils.basic

from torchvision.models.resnet import resnet18
from efficientnet_pytorch import EfficientNet

EPS = 1e-4

from functools import partial

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum

class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip

class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, predict_future_flow):
        super().__init__()
        backbone = resnet18(pretrained=False, zero_init_residual=True)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.predict_future_flow = predict_future_flow

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.feat_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=1, padding=0),
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

    def forward(self, x, bev_flip_indices=None):
        b, c, h, w = x.shape

        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2]) # note [-2] instead of [-3], since Y is gone now
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        feat_output = self.feat_head(x)
        segmentation_output = self.segmentation_head(x)
        instance_center_output = self.instance_center_head(x)
        instance_offset_output = self.instance_offset_head(x)
        instance_future_output = self.instance_future_head(x) if self.predict_future_flow else None

        return {
            'raw_feat': x,
            'feat': feat_output.view(b, *feat_output.shape[1:]),
            'segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),
            'instance_center': instance_center_output.view(b, *instance_center_output.shape[1:]),
            'instance_offset': instance_offset_output.view(b, *instance_offset_output.shape[1:]),
            'instance_flow': instance_future_output.view(b, *instance_future_output.shape[1:])
            if instance_future_output is not None else None,
        }

import torchvision
class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

class Encoder_eff(nn.Module):
    def __init__(self, C, version='b4'):
        super().__init__()
        self.C = C
        self.downsample = 8
        self.version = version

        if self.version == 'b0':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif self.version == 'b4':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']
        # print('input_1', input_1.shape)
        # print('input_2', input_2.shape)
        x = self.upsampling_layer(input_1, input_2)
        # print('x', x.shape)
        return x

    def forward(self, x):
        x = self.get_features(x)  # get feature vector
        x = self.depth_layer(x)  # feature and depth head
        return x

class SparseInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(SparseInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Initialize running statistics for inference
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))

    def forward(self, x):
        with torch.no_grad():
            is_non_zero = x.abs() > self.eps

        if self.training:
            with torch.no_grad():
                # Calculate mean and std only for non-zero elements
                num_non_zero = is_non_zero.float().sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
                mean = x.sum(dim=(2, 3), keepdim=True) / num_non_zero
                var = (x - mean).square().sum(dim=(2, 3), keepdim=True) / num_non_zero

                # Update running statistics during training
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.mean(dim=0, keepdim=True)
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.mean(dim=0, keepdim=True)
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize using mean and std
        x = torch.where(is_non_zero, (x - mean) / torch.sqrt(var + self.eps), torch.zeros([], device=x.device).expand_as(x))

        return x

class TinyUNet(nn.Module):
    def __init__(self, in_channels, out_channels_list=None, base_channels=32):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=13, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ) # (C, 224, 224)

        self.down_conv1 = self.make_down_conv(base_channels*2) # (2C, 112, 112)
        self.down_conv2 = self.make_down_conv(base_channels*4) # (4C, 56, 56)
        self.down_conv3 = self.make_down_conv(base_channels*8) # (8C, 28, 28)
        self.down_conv4 = self.make_down_conv(base_channels*16) # (16C, 14, 14)
        self.down_conv5 = self.make_down_conv(base_channels*32) # (32C, 7, 7)

        self.up_conv1 = self.make_up_conv(base_channels*16) # (16C, 14, 14)
        self.up_conv2 = self.make_up_conv(base_channels*8) # (16C, 28, 28)
        self.up_conv3 = self.make_up_conv(base_channels*4) # (16C, 56, 56)
        self.up_conv4 = self.make_up_conv(base_channels*2) # (16C, 112, 112)
        self.up_conv5 = self.make_up_conv(base_channels) # (16C, 224, 224)

        if out_channels_list is None:
            self.out_convs = None
        else:
            self.out_convs = nn.ModuleList([nn.Conv2d(base_channels * (2**level), out_channels, 1) for level, out_channels in zip(range(5,-1,-1), out_channels_list)])

    def make_down_conv(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels//2, channels, 4, stride=2, padding=1, bias=False),
            # nn.InstanceNorm2d(channels),
            SparseInstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def make_up_conv(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels*3, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            # SparseInstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def up_conv(self, name, x, skip_x):
        x = torch.cat([F.interpolate(x, size=skip_x.size(-1), mode='bilinear'), skip_x], dim=1)
        x = getattr(self, f'up_conv{name}')(x)
        return x

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down_conv1(x0)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)
        x4 = self.down_conv4(x3)
        x5 = self.down_conv5(x4)

        x6 = self.up_conv(1, x5, x4)
        x7 = self.up_conv(2, x6, x3)
        x8 = self.up_conv(3, x7, x2)
        x9 = self.up_conv(4, x8, x1)
        x10 = self.up_conv(5, x9, x0)

        return x10[..., 12:-12, 12:-12]

class KalmanFuser(nn.Module):
    def __init__(self, Y, feat2d_dim=128, base_channels=32):
        super().__init__()

        self.base_channels = base_channels
        self.feat2d_dim = feat2d_dim
        # self.log_epsilon = -10
        self.Y = Y

        self.radar_feature_extractor = TinyUNet(16 * Y, base_channels=base_channels)
        # self.camera_feature_extractor = nn.Conv2d(self.feat2d_dim*Y, base_channels, 1)
        self.register_buffer('camera_feature_extractor', torch.randn(base_channels, self.feat2d_dim*Y, 1, 1) * 0.05)
        # self.feat_to_init_state = nn.Conv2d(base_channels, base_channels*2, 1)
        # self.init_mats = nn.Parameter(torch.randn(1, base_channels*2, 1, 1, requires_grad=True))
        self.feat_to_mats = nn.Conv2d(base_channels, base_channels*2 + base_channels*4, 1)
        self.feat_to_mats_camera = nn.Conv2d(base_channels, base_channels*2, 1)
        self.z_to_radar = nn.Conv2d(base_channels, 16 * Y, 1)

        for conv in [self.feat_to_mats, self.feat_to_mats_camera]:
            conv.weight.data.normal_(std=1e-6)

    def forward(self, feat_bev_, metarad_occ_mem0):
        z_posteriors = []
        z_priors = []
        radar_mses = []

        nsweeps = 3

        for step in range(nsweeps):
            with torch.no_grad():
                mask = (metarad_occ_mem0[:,-1:] == (nsweeps - step))
                radar_raw = (metarad_occ_mem0[:,:-1] * mask.float()).permute(0,1,3,2,4)
                radar_raw = radar_raw.reshape(radar_raw.shape[0], -1, *radar_raw.shape[-2:])
                mask = (torch.ones(1, device=metarad_occ_mem0.device, dtype=torch.bool).expand_as(metarad_occ_mem0[:,:-1]) & mask).permute(0,1,3,2,4)
                mask = mask.reshape(mask.shape[0], -1, *mask.shape[-2:])
            radar_feat = self.radar_feature_extractor(radar_raw)

            if step == nsweeps-1:
                camera = feat_bev_
                camera_feat = F.conv2d(camera, self.camera_feature_extractor)

            if step == 0:
                s_init = (None, None)
                _, _, F_curr, H_curr, logQ_curr, logR_curr = self.feat_to_mats(torch.zeros_like(radar_feat[..., :1, :1])).split([self.base_channels] * 6, dim=1)
                F_curr = 1 + F_curr
                mu = 0
                var = 1
                
            pred_mu = F_curr * mu
            pred_var = F_curr.square() * var + logQ_curr.exp()
            res_var = H_curr.square() * pred_var + logR_curr.exp()
            z_pred = H_curr * pred_mu
            z_priors.append((z_pred, res_var.log()))

            z_mean, z_logvar, F_next, H_next, logQ_next, logR_next = self.feat_to_mats(radar_feat).split([self.base_channels] * 6, dim=1)
            F_next = 1 + F_next
            z_posteriors.append((z_mean, z_logvar))

            z_curr = torch.normal(z_mean, (0.5*z_logvar).exp()) if self.training else z_mean
            radar_mses.append(
                ((self.z_to_radar(z_curr) - radar_raw) * mask).square().mean(dim=(-1, -2)).view(-1, 16, self.Y)
            )

            residual = z_curr - z_pred
            kalman_gain = pred_var * H_curr / res_var
            mu = pred_mu + kalman_gain * residual
            var = (1 - kalman_gain * H_curr) * pred_var

            if step < nsweeps-1:
                F_curr, H_curr, logQ_curr, logR_curr = F_next, H_next, logQ_next, logR_next
            else:
                H_cam_curr, logR_cam_curr = self.feat_to_mats_camera(radar_feat).split([self.base_channels] * 2, dim=1)

                camera_pred = H_cam_curr * mu
                residual = camera_feat - camera_pred
                res_var = H_cam_curr.square() * var + logR_cam_curr.exp()
                camera_nll = residual.square()/(2*res_var)

                kalman_gain = var * H_cam_curr / res_var
                mu = mu + kalman_gain * residual
                var = (1 - kalman_gain * H_curr) * var

        sample = torch.normal(mu, (0.5*var).exp()) if self.training else mu

        return sample, z_posteriors, z_priors, radar_mses, camera_nll, s_init


class Segnet(nn.Module):
    def __init__(self, Z, Y, X, vox_util=None,
                 use_radar=False,
                 use_lidar=False,
                 use_metaradar=False,
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=128,
                 encoder_type="res101"):
        super(Segnet, self).__init__()
        assert (encoder_type in ["res101", "res50", "effb0", "effb4"])

        self.Z, self.Y, self.X = Z, Y, X
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.use_metaradar = use_metaradar
        self.do_rgbcompress = do_rgbcompress
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float().cuda()

        # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        if encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim)
        elif encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim)
        elif encoder_type == "effb0":
            self.encoder = Encoder_eff(feat2d_dim, version='b0')
        else:
            # effb4
            self.encoder = Encoder_eff(feat2d_dim, version='b4')

        # BEV compressor
        if self.use_radar:
            if self.use_metaradar:
                # self.bev_compressor = nn.Sequential(
                #     nn.Conv2d(feat2d_dim*Y + 16*Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                #     nn.InstanceNorm2d(latent_dim),
                #     nn.GELU(),
                # )
                self.kalman_fuser = KalmanFuser(Y, feat2d_dim=self.feat2d_dim, base_channels=32)
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y + self.kalman_fuser.base_channels, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y+1, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
        elif self.use_lidar:
            self.bev_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim*Y+Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
        else:
            if self.do_rgbcompress:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                # use simple sum
                pass

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=1,
            predict_future_flow=False
        )

        # Weights
        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.radar_recon_weights = nn.Parameter(torch.zeros(1, 16, 1, requires_grad=True))

        # set_bn_momentum(self, 0.1)

        if vox_util is not None:
            self.xyz_memA = utils.basic.gridcloud3d(1, Z, Y, X, norm=False)
            self.xyz_camA = vox_util.Mem2Ref(self.xyz_memA, Z, Y, X, assert_cube=False)
        else:
            self.xyz_camA = None

    def forward(self, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util, rad_occ_mem0=None):
        '''
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_camXs: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cam0_T_camXs: (B,S,4,4)
        vox_util: vox util object
        rad_occ_mem0:
            - None when use_radar = False, use_lidar = False
            - (B, 1, Z, Y, X) when use_radar = True, use_metaradar = False
            - (B, 16, Z, Y, X) when use_radar = True, use_metaradar = True
            - (B, 1, Z, Y, X) when use_lidar = True
        '''
        B, S, C, H, W = rgb_camXs.shape
        assert(C==3)
        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_camXs_ = __p(rgb_camXs)
        pix_T_cams_ = __p(pix_T_cams)
        cam0_T_camXs_ = __p(cam0_T_camXs)
        camXs_T_cam0_ = utils.geom.safe_inverse(cam0_T_camXs_)

        # rgb encoder
        device = rgb_camXs_.device
        rgb_camXs_ = (rgb_camXs_ + 0.5 - self.mean.to(device)) / self.std.to(device)
        if self.rand_flip:
            B0, _, _, _ = rgb_camXs_.shape
            self.rgb_flip_index = np.random.choice([0,1], B0).astype(bool)
            rgb_camXs_[self.rgb_flip_index] = torch.flip(rgb_camXs_[self.rgb_flip_index], [-1])
        feat_camXs_ = self.encoder(rgb_camXs_)
        if self.rand_flip:
            feat_camXs_[self.rgb_flip_index] = torch.flip(feat_camXs_[self.rgb_flip_index], [-1])
        _, C, Hf, Wf = feat_camXs_.shape

        sy = Hf/float(H)
        sx = Wf/float(W)
        Z, Y, X = self.Z, self.Y, self.X

        # unproject image feature to 3d grid
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)
        if self.xyz_camA is not None:
            xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B*S,1,1)
        else:
            xyz_camA = None
        feat_mems_ = vox_util.unproject_image_to_mem(
            feat_camXs_,
            utils.basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, Z, Y, X,
            xyz_camA=xyz_camA)
        feat_mems = __u(feat_mems_) # B, S, C, Z, Y, X

        # mask_mems = (torch.abs(feat_mems) > 0).float()
        # feat_mem = utils.basic.reduce_masked_mean(feat_mems, mask_mems, dim=1) # B, C, Z, Y, X
        with torch.no_grad():
            mask_mems = (torch.abs(feat_mems.detach()) > 0).float()
            mask_mems = mask_mems.sum(dim=1)
            mask_mems += 1e-6
        feat_mem = feat_mems.sum(dim=1) / mask_mems # B, C, Z, Y, X

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0,1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0,1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

            if rad_occ_mem0 is not None:
                rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
                rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])

        # bev compressing
        if self.use_radar:
            assert(rad_occ_mem0 is not None)
            if not self.use_metaradar:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                rad_bev = torch.sum(rad_occ_mem0, 3).clamp(0,1) # squish the vertical dim
                feat_bev_ = torch.cat([feat_bev_, rad_bev], dim=1)
                feat_bev = self.bev_compressor(feat_bev_)
            else:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                rad_bev_, z_posteriors, z_priors, radar_mses, camera_nll, s_init = self.kalman_fuser(feat_bev_, rad_occ_mem0)
                feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
                feat_bev = self.bev_compressor(feat_bev_)
        elif self.use_lidar:
            assert(rad_occ_mem0 is not None)
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
            rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, Y, Z, X)
            feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
            feat_bev = self.bev_compressor(feat_bev_)
        else: # rgb only
            if self.do_rgbcompress:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                feat_bev = self.bev_compressor(feat_bev_)
            else:
                feat_bev = torch.sum(feat_mem, dim=3)

        # bev decoder
        out_dict = self.decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)

        raw_e = out_dict['raw_feat']
        feat_e = out_dict['feat']
        seg_e = out_dict['segmentation']
        center_e = out_dict['instance_center']
        offset_e = out_dict['instance_offset']

        return raw_e, feat_e, seg_e, center_e, offset_e, z_posteriors, z_priors, radar_mses, camera_nll, s_init

