import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from sklearn.decomposition import PCA
import scipy.stats as sps
import numpy as np
from copy import deepcopy

from models.StyleGAN2.model import ModulatedConv2dPatchedFixedBasisDelta
from models.StyleGAN2.model import ModulatedConv2dPatchedSVDBasisDelta


def get_conv_from_generator(generator, conv_ix):
    if 'StyleGAN2Wrapper' in generator.__class__.__name__:
        return generator.style_gan2.convs[conv_ix].conv
    else:
        assert NotImplementedError


class WeightDeformatorFixedBasis(nn.Module):
    def __init__(self, generator, conv_layer_ix, directions_count,
                 basis_vectors=None, basis_vectors_path=None):
        super(WeightDeformatorFixedBasis, self).__init__()
        assert (basis_vectors is not None) or (basis_vectors_path is not None),\
            'either basis tensor or basis tensor path must be provided'

        # Get conv layer to be hooked
        # List is used for this layer not to show up in .parameters()
        if basis_vectors is None:
            basis_vectors = torch.load(basis_vectors_path)
        generator.style_gan2.convs[conv_layer_ix].conv = ModulatedConv2dPatchedFixedBasisDelta(
            basis_vectors=basis_vectors.cuda(),
            conv_to_patch=generator.style_gan2.convs[conv_layer_ix].conv,
            directions_count=directions_count
        )

        self.hooked_conv_layer = [get_conv_from_generator(generator, conv_layer_ix)]
        self.disable_deformation()

    def deformate(self, batch_directions, batch_shifts):
        self.hooked_conv_layer[0].is_deformated = True
        self.hooked_conv_layer[0].batch_directions = batch_directions
        self.hooked_conv_layer[0].batch_shifts = batch_shifts

    def disable_deformation(self):
        self.hooked_conv_layer[0].is_deformated = False

    def parameters(self):
        return [self.hooked_conv_layer[0].direction_to_basis_coefs]


class WeightDeformatorSVDBasis(nn.Module):
    def __init__(self, generator, conv_layer_ix, directions_count):
        super(WeightDeformatorSVDBasis, self).__init__()
        
        # Get conv layer to be hooked
        # List is used for this layer not to show up in .parameters()
        generator.style_gan2.convs[conv_layer_ix].conv = ModulatedConv2dPatchedSVDBasisDelta(
            conv_to_patch=generator.style_gan2.convs[conv_layer_ix].conv,
            directions_count=directions_count
        )

        self.hooked_conv_layer = [get_conv_from_generator(generator, conv_layer_ix)]

        self.disable_deformation()

    def deformate(self, batch_directions, batch_shifts):
        self.hooked_conv_layer[0].is_deformated = True
        self.hooked_conv_layer[0].batch_directions = batch_directions
        self.hooked_conv_layer[0].batch_shifts = batch_shifts

    def disable_deformation(self):
        self.hooked_conv_layer[0].is_deformated = False

    def parameters(self):
        return [self.hooked_conv_layer[0].direction_to_eigenvalues_delta]
