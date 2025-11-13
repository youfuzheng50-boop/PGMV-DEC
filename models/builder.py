
from .Resnet50 import TimmCNNEncoder
from utils.transform_utils import get_eval_transforms
from utils.constants import MODEL2CONSTANTS

def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))

    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size=target_img_size)

    return model

def ht_get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))

    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size=target_img_size)

    return model, img_transforms