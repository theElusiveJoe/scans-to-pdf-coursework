import torch
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

# import torchvision
# from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
# from torchvision.models.resnet import resnet50
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


keys = '''backbone.fpn.inner_blocks.0.weight
backbone.fpn.inner_blocks.0.bias
backbone.fpn.inner_blocks.1.weight
backbone.fpn.inner_blocks.1.bias
backbone.fpn.inner_blocks.2.weight
backbone.fpn.inner_blocks.2.bias
backbone.fpn.inner_blocks.3.weight
backbone.fpn.inner_blocks.3.bias
backbone.fpn.layer_blocks.0.weight
backbone.fpn.layer_blocks.0.bias
backbone.fpn.layer_blocks.1.weight
backbone.fpn.layer_blocks.1.bias
backbone.fpn.layer_blocks.2.weight
backbone.fpn.layer_blocks.2.bias
backbone.fpn.layer_blocks.3.weight
backbone.fpn.layer_blocks.3.bias
rpn.head.conv.weight
rpn.head.conv.bias'''.strip().split('\n')

vals = '''backbone.fpn.inner_blocks.0.0.weight
backbone.fpn.inner_blocks.0.0.bias
backbone.fpn.inner_blocks.1.0.weight
backbone.fpn.inner_blocks.1.0.bias
backbone.fpn.inner_blocks.2.0.weight
backbone.fpn.inner_blocks.2.0.bias
backbone.fpn.inner_blocks.3.0.weight
backbone.fpn.inner_blocks.3.0.bias
backbone.fpn.layer_blocks.0.0.weight
backbone.fpn.layer_blocks.0.0.bias
backbone.fpn.layer_blocks.1.0.weight
backbone.fpn.layer_blocks.1.0.bias
backbone.fpn.layer_blocks.2.0.weight
backbone.fpn.layer_blocks.2.0.bias
backbone.fpn.layer_blocks.3.0.weight
backbone.fpn.layer_blocks.3.0.bias
rpn.head.conv.0.0.weight
rpn.head.conv.0.0.bias
'''.strip().split('\n')
modules_mapping = dict([(x,y) for x, y in zip(keys, vals)])


def create_model(config, mode='raw', name=None):
    assert mode in ['raw', 'publaynet', 'custom']

    model = fasterrcnn_resnet50_fpn(num_classes=config.CLASSES_NUM)

    if mode == 'raw':
        print('raw model created')
        pass

    elif mode == 'publaynet':
        pretrained_states_dict = torch.load(config.TRAINED_MODELS_PATH.joinpath('mask_rcnn_publaynet.pth'))['model']

        for x, y in modules_mapping.items():
            if x not in model.state_dict().keys():
                pretrained_states_dict[y] = pretrained_states_dict[x]

        keys = list(pretrained_states_dict.keys())
        for key in keys:
            if key not in model.state_dict().keys():
                pretrained_states_dict.pop(key)

        model.load_state_dict(pretrained_states_dict)
        print('publaynet pretrained model loaded')

    elif mode == 'custom':
        model.load_state_dict(torch.load(config.TRAINED_MODELS_PATH.joinpath(f'{name}.pth')))
        print('custom state model loaded')

    return model