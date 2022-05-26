from models.resnet_26 import ResNetCifar as ResNet
from torch import nn


def build_model(args):
    print('Building model...')
    if args.dataset == 'cifar10':
        classes = 10
    elif args.dataset == "cifar100":
        classes = 100

    if args.group_norm == 0:
        norm_layer = nn.BatchNorm2d
    else:
        def gn_helper(planes):
            return nn.GroupNorm(args.group_norm, planes)
        norm_layer = gn_helper

    if hasattr(args, 'detach') and args.detach:
        detach = args.shared
    else:
        detach = None

    net = ResNet(args.depth, args.width, channels=3, classes=classes, norm_layer=norm_layer, detach=detach).cuda()

    return net