from torch import nn
import resnet
import torch.utils.model_zoo as model_zoo

_IMAGENET_URLS = {
    'resnet18':  'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34':  'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50':  'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# Keys that differ between ImageNet pretrain and our model (always excluded).
_EXCLUDE = {'conv1.weight', 'fc.weight', 'fc.bias'}


def build_model(model_name, pretrained=False):
    """
    Build a ResNetCDPC model.  pretrained=True loads ImageNet backbone
    weights into the ResNet trunk (conv2-layer4) while leaving the new
    CDPC heads and the 2-channel conv1 randomly initialised.
    """
    builders = {
        'resnet18':  resnet.resnet18_cdpc,
        'resnet34':  resnet.resnet34_cdpc,
        'resnet50':  resnet.resnet50_cdpc,
        'resnet101': resnet.resnet101_cdpc,
    }
    if model_name not in builders:
        raise ValueError('Unsupported model: {}. Choose from {}'.format(
            model_name, list(builders.keys())))

    model = builders[model_name]()

    if pretrained and model_name in _IMAGENET_URLS:
        pretrained_dict = model_zoo.load_url(_IMAGENET_URLS[model_name])
        model_dict = model.state_dict()
        filtered = {k: v for k, v in pretrained_dict.items()
                    if k in model_dict and k not in _EXCLUDE}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print('Loaded {}/{} ImageNet weights into backbone.'.format(
            len(filtered), len(model_dict)))

    return model
