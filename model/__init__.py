from .alexnet import AlexNet
from .lenet import LeNet
from .vgg import VggNet
from .googlenet import GoogLeNet
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .densenet import densenet121, densenet161, densenet169, densenet201
from .mobilenetv2 import MobileNetV2
from .mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from .shufflenetv2 import shufflenet_v2_x1_0, shufflenet_v2_x0_5
from .efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
from .convnetxt import convnext_tiny, convnext_base, convnext_large, convnext_small, convnext_xlarge
from .loss_optim_scheduler import *


def get_model(s):
    return {"lenet": LeNet,
            "alexnet": AlexNet,
            "vgg": VggNet,
            "googlenet": GoogLeNet,
            'resnet18': resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
            "resnet101": resnet101,
            'resnet152': resnet152,
            'densenet121': densenet121,
            'densenet161': densenet161,
            'densenet169': densenet169,
            'densenet201': densenet201,
            'mobilenetv2': MobileNetV2,
            'mobilenetv3_small': mobilenet_v3_small,
            'mobilenetv3_large': mobilenet_v3_large,
            'shufflenetv2_x0_5': shufflenet_v2_x0_5,
            'shufflenetv2_x1_0': shufflenet_v2_x1_0,
            'efficientnet_b0': efficientnet_b0,
            'efficientnet_b1': efficientnet_b1,
            'convnext_tiny': convnext_tiny,
            'convnext_base': convnext_base,
            'convnext_large': convnext_large,
            'convnext_small': convnext_small,
            }[s.lower()]


def get_loss(s):
    return {
        'l1': l1,
        'l2': l2,
        'bce': bce,
        'ce': ce
    }[s.lower()]


def get_optim(s):
    return {
        'adam': adam,
        'sgd': sgd,
        'adagrad': adagrad,
        'rmsprop': rmsprop,
    }[s.lower()]


def get_scheduler(s):
    return {
        'steplr': steplr,
        'multisteplr': multisteplr,
        'cosineannealinglr': cosineannealinglr,
        'reducelronplateau': reducelronplateau,
        'lambdalr': lambdalr,
        'cycliclr': cycliclr,
    }[s.lower()]
