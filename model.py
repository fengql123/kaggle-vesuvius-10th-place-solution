import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead
import timm
from resnet3d import generate_model
from config import CFG


# poolers and decoders
class VolumeProjection(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.fc = torch.nn.Linear(int(np.prod(input_shape)), int(np.prod(output_shape)))
        self.flatten = torch.nn.Flatten(start_dim=2)
        self.output_shape = output_shape

    def forward(self, x):
        y = self.flatten(x)
        y = self.fc(y)
        y = y.view((y.shape[0], y.shape[1], self.output_shape[0], self.output_shape[1]))
        return y


class AttentionPool(torch.nn.Module):
    def __init__(self, depth, height, width):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.ones(1, 1, depth, height, width))
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # Apply softmax along the depth dimension to obtain attention weights
        attention_weights = self.softmax(self.attention_weights)
        # Perform attention pooling by multiplying the attention weights with the input tensor
        pooled_output = torch.mul(attention_weights, x)
        # Sum the pooled output along the depth dimension
        pooled_output = torch.sum(pooled_output, dim=2)
        return pooled_output


class Subvolume3DcnnEncoder(nn.Module):
    def __init__(self, batch_norm_momentum, filters):
        super().__init__()
        strides = [1, 2, 2, 2]
        filter_sizes = [1] + filters
        filter_list_pairs = list(zip(filter_sizes[:-1], filter_sizes[1:]))  # [(1, 16), (16, 32), (32, 64), (64, 128)]
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv3d(chan_in, chan_out, kernel_size=3, stride=stride, padding=1),
                nn.ReLU(),
                nn.BatchNorm3d(num_features=filter_, momentum=batch_norm_momentum)
            )
                for (chan_in, chan_out), stride, filter_ in zip(filter_list_pairs, strides, filters)])
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.conv_layers(x)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool3d(x.clamp(min=eps).pow(p), (x.size(-3), x.size(-2), x.size(-1))).pow(1./p)


class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=encoder_dims[i] + encoder_dims[i - 1],
                    out_channels=encoder_dims[i - 1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(encoder_dims[i - 1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            f_down = self.convs[i - 1](f)
            feature_maps[i - 1] = f_down

        mask = self.logit(feature_maps[0])
        mask = self.up(mask)
        return mask


class Decoder3D(nn.Module):
    def __init__(self, cfg, encoder_dims):
        super().__init__()
        self.cfg = cfg
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(
                    in_channels=encoder_dims[i] + encoder_dims[i - 1],
                    out_channels=encoder_dims[i - 1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm3d(encoder_dims[i - 1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv3d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(size=(cfg.in_chans, cfg.size, cfg.size), mode="trilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], size=feature_maps[i - 1].shape[2:], mode="trilinear")
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            f_down = self.convs[i - 1](f)
            feature_maps[i - 1] = f_down
            
        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask


# 2.5D
class CustomModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        if cfg.use_pool:
            in_chans = 1
        else:
            in_chans = cfg.in_chans
            
        self.encoder = timm.create_model(
            cfg.encoder,
            in_chans=in_chans,
            features_only=True,
            drop_rate=cfg.drop_rate,
            drop_path_rate=cfg.drop_path_rate,
            out_indices=tuple(range(cfg.depth)),
            pretrained=True
        )

        temp = self.encoder(torch.rand((1, 1, cfg.size, cfg.size)))
        
        encoder_channels = [in_chans] + self.encoder.feature_info.channels()
        decoder_channels = cfg.decoder_channels
        
        if cfg.use_pool:
            self.pooler = nn.ModuleList([AttentionPool(self.cfg.in_chans, x.shape[-2], x.shape[-1]) for x in temp])

        if cfg.decoder == "Unet":
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels,
                decoder_channels=decoder_channels,
                n_blocks=cfg.depth,
            )
            self.segmentation_head = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=cfg.target_size,
                activation=None,
                kernel_size=3,
            )
        elif cfg.decoder == "Unet++":
            self.decoder = smp.decoders.unetplusplus.decoder.UnetPlusPlusDecoder(
                encoder_channels=encoder_channels,
                decoder_channels=decoder_channels,
                n_blocks=cfg.depth,
            )
            self.segmentation_head = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=cfg.target_size,
                activation=None,
                kernel_size=3,
            )
        elif cfg.decoder == "DeepLabV3+":
            self.decoder = smp.decoders.deeplabv3.decoder.DeepLabV3PlusDecoder(
                encoder_channels=encoder_channels[:cfg.depth + 1],
            )
            self.segmentation_head = SegmentationHead(
                in_channels=self.decoder.out_channels,
                out_channels=cfg.target_size,
                activation=None,
                kernel_size=1,
                upsampling=4,
            )
            
        if cfg.use_denoiser:
            self.denoiser = smp.Unet(
                encoder_name="tu-resnet10t", # "tu-resnet10t" "resnet18"
                encoder_weights="imagenet",
                in_channels=1,
                classes=1,
                activation=None,
            )

    def get_features(self, x):
        feat_maps = self.encoder(x)
        return feat_maps

    def forward(self, x):
        if self.cfg.use_pool:
            bs = x.shape[0]
            x = x.view((-1, 1, x.shape[-2], x.shape[-1]))
            feat_maps = self.get_features(x)
            feat_maps_pooled = []
            for i, f in enumerate(feat_maps):
                f = f.view((bs, -1, self.cfg.in_chans, f.shape[-2], f.shape[-1]))
                o = self.pooler[i](f)
                feat_maps_pooled.append(o)
        else:
            feat_maps_pooled = self.get_features(x)
        feat_maps_pooled = [x] + feat_maps_pooled
        decoder_output = self.decoder(*feat_maps_pooled)
        masks = self.segmentation_head(decoder_output)
        
        if self.cfg.use_denoiser:
            noise = self.denoiser(masks)
            masks = masks - noise
        
        return masks


# 3D-2D
class SegModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.encoder3d == "resnet":
            # https://github.com/kenshohara/3D-ResNets-PyTorch
            self.encoder = generate_model(model_depth=cfg.encoder_depth, n_input_channels=1)
            channels = [64, 128, 256, 512]

        temp = self.encoder(torch.rand((1, 1, cfg.in_chans, cfg.size, cfg.size)))
        
        if cfg.decoder3d2d == "Custom":
            self.decoder = Decoder(encoder_dims=channels, upscale=4)
        elif cfg.decoder3d2d == "FPN":
            self.decoder = smp.decoders.fpn.decoder.FPNDecoder(
                encoder_channels=channels,
                encoder_depth=4,
                pyramid_channels=256,
                segmentation_channels=128,
                dropout=0.2,
                merge_policy="add",
            )
            upscale = 4
            self.segmentation_head = SegmentationHead(
                in_channels=128,
                out_channels=cfg.target_size,
                activation=None,
                kernel_size=1,
                upsampling=upscale,
            )

        if cfg.pooler == "projection":
            self.poolers = nn.ModuleList([VolumeProjection(x.shape[2:], x.shape[3:]) for x in temp])
            
        elif cfg.pooler == "conv":
            self.poolers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(in_channels=x.shape[2] * x.shape[1], out_channels=x.shape[1], kernel_size=1),
                        nn.BatchNorm2d(x.shape[1]),
                        nn.ReLU(inplace=True)
                    ) for x in temp
                ]
            )
            
        elif cfg.pooler == "attention":
            self.poolers = nn.ModuleList([AttentionPool(x.shape[2], x.shape[3], x.shape[4]) for x in temp])

        elif cfg.pooler == "conv_attention":
            self.poolers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv3d(x.shape[1], x.shape[1], kernel_size=1),
                ) for x in temp
            ])
            
        if cfg.use_denoiser:
            self.denoiser = smp.Unet(
                encoder_name="tu-resnet10t", # "tu-resnet10t" "resnet18"
                encoder_weights="imagenet",
                in_channels=1,
                classes=1,
                activation=None,
            )

    def forward(self, x):
        feat_maps = self.encoder(x)

        if self.cfg.pooler == "projection" or self.cfg.pooler == "attention":
            feat_maps_pooled = []
            for i, x in enumerate(feat_maps):
                pooled = self.poolers[i](x)
                feat_maps_pooled.append(pooled)

        elif self.cfg.pooler == "conv":
            feat_maps_pooled = []
            for i, x in enumerate(feat_maps):
                x = x.view((x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
                pooled = self.poolers[i](x)
                feat_maps_pooled.append(pooled)

        elif self.cfg.pooler == "mean":
            feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]
            
        elif self.cfg.pooler == "max":
            feat_maps_pooled = [torch.amax(f, dim=2) for f in feat_maps]

        elif self.cfg.pooler == "conv_attention":
            feat_maps_pooled = []
            for i, x in enumerate(feat_maps):
                w = self.poolers[i](x)
                w = F.softmax(w, 2)
                pooled = (w * x).sum(2)
                feat_maps_pooled.append(pooled)
               
        if self.cfg.decoder3d2d == "Custom":        
            pred_mask = self.decoder(feat_maps_pooled)
        else:
            pred_mask = self.decoder(*feat_maps_pooled)
            pred_mask = self.segmentation_head(pred_mask)
        if self.cfg.use_denoiser:
            noise = self.denoiser(pred_mask)
            pred_mask = pred_mask - noise
        
        return pred_mask

    def load_pretrained_weights(self, state_dict):
        # Convert 3 channel weights to single channel
        # ref - https://timm.fast.ai/models#Case-1:-When-the-number-of-input-channels-is-1
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        self.encoder.load_state_dict(state_dict, strict=False)


# ink-classifier
class LinearInkDecoder(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.fc = nn.Linear(input_shape, 1)
        if cfg.feature_pooler == "mean":
            self.pool = nn.AdaptiveAvgPool3d(1)
        elif cfg.feature_pooler == "max":
            self.pool = nn.AdaptiveMaxPool3d(1)
    def forward(self, x):
        x = self.pool(x).squeeze(dim=-1).squeeze(dim=-1).squeeze(dim=-1)
        return self.fc(x)


class InkClassifier(nn.Module):
    def __init__(self, cfg, batch_norm_momentum=0.1, filters=[16, 32, 64, 128]):
        super().__init__()
        self.cfg = cfg
        if cfg.encoder3d == "custom":
            self.encoder = Subvolume3DcnnEncoder(batch_norm_momentum, filters)
            dim = filters[-1]
            self.decoder = LinearInkDecoder(cfg, dim)
        elif cfg.encoder3d == "resnet":
            self.encoder = generate_model(model_depth=CFG.encoder_depth, n_input_channels=1)
            dim = 512
            if cfg.feature_pooler in ["mean", "max"]:
                self.decoder = LinearInkDecoder(cfg, dim)
            
            elif cfg.feature_pooler == "gem":
                self.pool = nn.Sequential(
                    GeM(),
                    nn.Flatten(),
                )
                temp = self.pool(self.encoder(torch.rand((1, 1, cfg.in_chans, cfg.region_size, cfg.region_size)))[-1])
                dim = temp.shape[-1]
                self.decoder = nn.Sequential(
                    GeM(),
                    nn.Flatten(),
                    nn.Linear(dim, cfg.target_size),
                )
                
            elif cfg.feature_pooler == "None":
                self.pool = nn.Flatten()
                temp = self.pool(self.encoder(torch.rand((1, 1, CFG.in_chans, cfg.region_size, cfg.region_size)))[-1])
                dim = temp.shape[-1]
                self.decoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(dim, cfg.target_size),
                )
        elif cfg.encoder3d == "2D":
            self.encoder = timm.create_model(
                cfg.encoder,
                in_chans=1,
                features_only=True,
                drop_rate=cfg.drop_rate,
                drop_path_rate=cfg.drop_path_rate,
                out_indices=tuple(range(cfg.depth)),
                pretrained=True
            )
            dim = self.encoder(torch.rand(1, 1, cfg.region_size, cfg.region_size))[-1].shape[1]
            self.decoder = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(dim * cfg.in_chans, cfg.target_size)
                

    def forward(self, x):
        if self.cfg.encoder3d == "custom":
            x = self.encoder(x)
            return self.decoder(x)
        elif self.cfg.encoder3d == "resnet":
            x = self.encoder(x)[-1]
            return self.decoder(x)
        elif self.cfg.encoder3d == "2D":
            bs = x.shape[0]
            x = x.view(-1, 1, self.cfg.region_size, self.cfg.region_size)
            x = self.encoder(x)[-1]
            x = self.decoder(x)
            x = x.squeeze(-1).squeeze(-1)
            x = x.view(bs, -1)
            x = self.fc(x)
            return x

    def load_pretrained_weights(self, state_dict):
        # Convert 3 channel weights to single channel
        # ref - https://timm.fast.ai/models#Case-1:-When-the-number-of-input-channels-is-1
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        self.encoder.load_state_dict(state_dict, strict=False)
        
        
class SegModelV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = generate_model(model_depth=cfg.encoder_depth, n_input_channels=1)
        self.decoder = Decoder3D(cfg, encoder_dims=[64, 128, 256, 512])
            
        if cfg.pooler3d == "attention":
            self.pooler = AttentionPool(cfg.in_chans, cfg.size, cfg.size)
        elif cfg.pooler3d == "conv_attention":
            self.pooler = nn.Sequential(
                nn.Conv3d(1, 1, kernel_size=1),
            )
        elif cfg.pooler3d == "unet":
            self.pooler = smp.Unet(
                encoder_name="resnet18", # "tu-resnet10t" "resnet18"
                encoder_weights="imagenet",
                in_channels=cfg.in_chans,
                classes=1,
                activation=None,
            )
        if cfg.use_denoiser:
            self.denoiser = smp.Unet(
                encoder_name="tu-resnet10t", # "tu-resnet10t" "resnet18"
                encoder_weights="imagenet",
                in_channels=1,
                classes=1,
                activation=None,
            )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
            
        if self.cfg.pooler3d == "mean":
            x = torch.mean(x, dim=2)
        elif self.cfg.pooler3d == "max":
            x = torch.amax(x, dim=2)
        elif self.cfg.pooler3d == "attention":
            x = self.pooler(x)
        elif self.cfg.pooler3d == "conv_attention":
            w = self.pooler(x)
            w = F.softmax(w, dim=2)
            x = (w * x).sum(dim=2)
        elif self.cfg.pooler3d == "unet":
            x = x.squeeze(dim=1)
            x = self.pooler(x)
        elif self.cfg.pooler3d == "cls":
            x = x[:,:,0,:,:].squeeze(dim=2)
            
        if self.cfg.use_denoiser:
            noise = self.denoiser(x)
            x = x - noise
        return x
    
    def load_pretrained_weights(self, state_dict):
        # Convert 3 channel weights to single channel
        # ref - https://timm.fast.ai/models#Case-1:-When-the-number-of-input-channels-is-1
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        self.encoder.load_state_dict(state_dict, strict=False)
        

def build_model(cfg, logger):
    logger.info(f'model: {cfg.model}')
    if cfg.model == "2.5D":
        logger.info(f'encoder: {cfg.encoder}')
        logger.info(f'decoder: {cfg.decoder}')
        model = CustomModel(cfg)
    elif cfg.model == "3D-2D":
        model = SegModel(cfg)
        model.load_pretrained_weights(torch.load(cfg.weight_path3d)["state_dict"])
    elif cfg.model == "ink-classifier":
        model = InkClassifier(cfg, filters=[16, 32, 64, 128])
        if cfg.encoder3d == "resnet":
            model.load_pretrained_weights(torch.load(cfg.weight_path3d)["state_dict"])
    elif cfg.model == "3D-3D-1D":
        model = SegModelV2(cfg)
        model.load_pretrained_weights(torch.load(cfg.weight_path3d)["state_dict"])
    return model
