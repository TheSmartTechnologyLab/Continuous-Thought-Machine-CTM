import torch
import torch.nn as nn
import os
import math
import warnings
# from models.modules import Identity

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity


        # activation = None
        # activation = out.detach().cpu().numpy()
        out = self.relu(out)
        # return out, activation

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        feature_scales,
        stride,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        do_initial_max_pool=True,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # NOTE: Important!
        # This has changed from a kernel size of 7 (padding=3) to a kernel of 3 (padding=1)
        # The reason for this was to limit the receptive field to constrain models to
        # "Looking around" to gather information.

        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        ) if in_channels in [1, 3] else nn.LazyConv2d(
            self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if do_initial_max_pool else Identity()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.feature_scales = feature_scales
        if 2 in feature_scales:
            self.layer2 = self._make_layer(
                block, 128, layers[1], stride=stride, dilate=replace_stride_with_dilation[0]
            )
            if 3 in feature_scales:
                self.layer3 = self._make_layer(
                    block, 256, layers[2], stride=stride, dilate=replace_stride_with_dilation[1]
                )
                if 4 in feature_scales:
                    self.layer4 = self._make_layer(
                        block, 512, layers[3], stride=stride, dilate=replace_stride_with_dilation[2]
                    )

        # NOTE: Commented this out as it is not used anymore for this work, kept it for reference
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        activations = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # if return_activations: activations.append(torch.clone(x))
        x = self.layer1(x)

        if 2 in self.feature_scales:
            x = self.layer2(x)
            if 3 in self.feature_scales:
                x = self.layer3(x)
                if 4 in self.feature_scales:
                    x = self.layer4(x)
        return x


def _resnet(in_channels, feature_scales, stride, arch, block, layers, pretrained, progress, device, do_initial_max_pool, **kwargs):
    model = ResNet(in_channels, feature_scales, stride, block, layers, do_initial_max_pool=do_initial_max_pool, **kwargs)
    if pretrained:
        assert in_channels==3
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + '/state_dicts/' + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(in_channels, feature_scales, stride=2, pretrained=False, progress=True, device="cpu", do_initial_max_pool=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(in_channels,
        feature_scales, stride, "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, do_initial_max_pool, **kwargs
    )


def resnet34(in_channels, feature_scales, stride=2, pretrained=False, progress=True, device="cpu", do_initial_max_pool=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(in_channels,
        feature_scales, stride, "resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, device, do_initial_max_pool, **kwargs
    )


def resnet50(in_channels, feature_scales, stride=2, pretrained=False, progress=True, device="cpu", do_initial_max_pool=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(in_channels,
        feature_scales, stride, "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, device, do_initial_max_pool, **kwargs
    )


def resnet101(in_channels, feature_scales, stride=2, pretrained=False, progress=True, device="cpu", do_initial_max_pool=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(in_channels,
        feature_scales, stride, "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, device, do_initial_max_pool, **kwargs
    )


def resnet152(in_channels, feature_scales, stride=2, pretrained=False, progress=True, device="cpu", do_initial_max_pool=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(in_channels,
        feature_scales, stride, "resnet152", Bottleneck, [3, 4, 36, 3], pretrained, progress, device, do_initial_max_pool, **kwargs
    )

def prepare_resnet_backbone(backbone_type):

    resnet_family = resnet18 # Default
    if '34' in backbone_type: resnet_family = resnet34
    if '50' in backbone_type: resnet_family = resnet50
    if '101' in backbone_type: resnet_family = resnet101
    if '152' in backbone_type: resnet_family = resnet152

    # Determine which ResNet blocks to keep
    block_num_str = backbone_type.split('-')[-1]
    hyper_blocks_to_keep = list(range(1, int(block_num_str) + 1)) if block_num_str.isdigit() else [1, 2, 3, 4]

    backbone = resnet_family(
        3,
        hyper_blocks_to_keep,
        stride=2,
        pretrained=False,
        progress=True,
        device="cpu",
        do_initial_max_pool=True,
    )

    return backbone


class TorchvisionBackboneWrapper(torch.nn.Module):
    """Simple wrapper for torchvision models to return spatial feature maps.

    This wrapper attempts to keep the convolutional feature extractor and remove
    classification heads. The exact behavior may vary per architecture; if the
    resulting feature shape cannot be reshaped, the wrapper will return the raw
    output and the CTM will try to use it (may require positional embedding 'none').
    """
    def __init__(self, name, pretrained=True, device='cpu'):
        super().__init__()
        import torchvision
        import torch
        import math
        import warnings
        import torch.nn as nn

        self.name = name
        self.pretrained = pretrained
        self.device = device

        # Load model from torchvision (via hub fallback)
        try:
            self.model = getattr(torchvision.models, name)(pretrained=pretrained)
        except Exception:
            # Try torch.hub if torchvision version doesn't expose constructor
            self.model = torch.hub.load('pytorch/vision:v0.10.0', name, pretrained=pretrained)

        # Remove classification heads where possible
        for attr in ('classifier', 'fc', 'head', 'avgpool'):
            if hasattr(self.model, attr):
                try:
                    setattr(self.model, attr, nn.Identity())
                except Exception:
                    pass

        self.model.to(device)

    def forward(self, x):
        out = self.model(x)
        # If output is flat, attempt heuristic reshape back to (B,C,H,W)
        if out.dim() == 2:
            B, F = out.shape
            # try to guess spatial dims based on common channel sizes
            for nc in (256, 512, 1024, 2048, 128, 64):
                if F % nc == 0:
                    wh = int(math.sqrt(F // nc))
                    if wh * wh * nc == F:
                        return out.reshape(B, nc, wh, wh)
            warnings.warn(f"Couldn't reshape flattened features from {self.name}; returning raw output.")
        return out