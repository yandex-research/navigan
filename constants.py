from enum import Enum

from latent_deformator import DeformatorType
from trainer import ShiftDistribution


class DeformatorTarget(Enum):
    LATENT = 1
    WEIGHT_SVD = 2
    WEIGHT_FIXEDBASIS = 3


HUMAN_ANNOTATION_FILE = 'human_annotation.txt'


DEFORMATOR_TYPE_DICT = {
    'fc': DeformatorType.FC,
    'linear': DeformatorType.LINEAR,
    'id': DeformatorType.ID,
    'ortho': DeformatorType.ORTHO,
    'proj': DeformatorType.PROJECTIVE,
    'random': DeformatorType.RANDOM,
}


DEFORMATOR_TARGET_DICT = {
    'latent': DeformatorTarget.LATENT,
    'weight_svd': DeformatorTarget.WEIGHT_SVD,
    'weight_fixedbasis': DeformatorTarget.WEIGHT_FIXEDBASIS
}


SHIFT_DISTRIDUTION_DICT = {
    'normal': ShiftDistribution.NORMAL,
    'uniform': ShiftDistribution.UNIFORM,
    None: None
}


WEIGHTS = {
    'BigGAN': 'models/pretrained/generators/BigGAN/G_ema.pth',
    'ProgGAN': 'models/pretrained/generators/ProgGAN/100_celeb_hq_network-snapshot-010403.pth',
    'SN_MNIST': 'models/pretrained/generators/SN_MNIST',
    'SN_Anime': 'models/pretrained/generators/SN_Anime',
    'StyleGAN2': 'models/pretrained/StyleGAN2/stylegan2-car-config-f.pt',
}
