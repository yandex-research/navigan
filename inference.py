from weight_deformator import WeightDeformatorFixedBasis
from weight_deformator import WeightDeformatorSVDBasis

import torch 
import torch.nn as nn
import tempfile
import os
import sys


class GeneratorWithWeightDeformator(nn.Module):
    def __init__(self, generator, deformator_type, layer_ix, **kwargs):
        super().__init__()

        assert deformator_type in ['svd', 'svd_rectification', 'hessian', 'hessian_rectification']

        if '_rectification' in deformator_type:
            assert 'checkpoint_path' in kwargs, 'checkpoint_path argument must be specified'

        if 'hessian' in deformator_type:
            assert 'eigenvectors_path' in kwargs, 'eigenvectors_path argument must be specified'

        self.G = generator
        self.deformator_type = deformator_type
        self.layer_ix = layer_ix

        if deformator_type == 'svd':
            self.__create_wd_svd(kwargs)
        elif deformator_type == 'svd_rectification':
            self.__create_wd_svd_rectification(kwargs)
        elif deformator_type == 'hessian':
            self.__create_wd_hessian(kwargs)
        elif deformator_type == 'hessian_rectification':
            self.__create_wd_hessian_rectification(kwargs)
        else:
            raise NotImplementedError

    def deformate(self, batch_directions, batch_shifts):
        self.wd.deformate(batch_directions, batch_shifts)

    def save_deformation(self, path, direction_ix, shift_scale=1.0):
        state_dict = {'layer_ix': self.layer_ix,}

        self.deformate(direction_ix, shift_scale)
        conv = self.G.style_gan2.convs[self.layer_ix].conv
        with torch.no_grad():
            state_dict['shift'] = conv.weight_shifts(1).cpu().unsqueeze(1)
        torch.save(state_dict, path)

    def forward(self, x):
        return self.G(x)

    def __create_wd_svd(self, kwargs):
        c_out, c_in, k_x, k_y = self.G.style_gan2.convs[self.layer_ix].conv.weight.shape[-4:]
        directions_count = min(k_x * k_y * c_in, c_out)

        self.wd = WeightDeformatorSVDBasis(
            generator=self.G,
            conv_layer_ix=self.layer_ix,
            directions_count=directions_count
        )

        self.G.style_gan2.convs[self.layer_ix].conv.direction_to_eigenvalues_delta = nn.Parameter(
            torch.eye(directions_count).cuda()
        )

        self.G = self.G.cuda()
        self.directions_count = directions_count

    def __create_wd_svd_rectification(self, kwargs):
        generator_dict = torch.load(kwargs['checkpoint_path'])['generator']
        dict_key_prefix = f'style_gan2.convs.{self.layer_ix}.conv.'

        direction_to_eigenvalues_delta = generator_dict[dict_key_prefix + 'direction_to_eigenvalues_delta']
        u = generator_dict[dict_key_prefix + 'u']
        vh = generator_dict[dict_key_prefix + 'vh']

        directions_count = direction_to_eigenvalues_delta.shape[0]

        self.wd = WeightDeformatorSVDBasis(
            generator=self.G,
            conv_layer_ix=self.layer_ix,
            directions_count=directions_count
        )

        self.G.style_gan2.convs[self.layer_ix].conv.direction_to_eigenvalues_delta = nn.Parameter(
            direction_to_eigenvalues_delta.cuda()
        )
        self.G.style_gan2.convs[self.layer_ix].conv.u = nn.Parameter(
            u.cuda()
        )
        self.G.style_gan2.convs[self.layer_ix].conv.vh = nn.Parameter(
            vh.cuda()
        )

        self.G = self.G.cuda()
        self.directions_count = directions_count

    def __create_wd_hessian(self, kwargs):
        directions_count = torch.load(kwargs['eigenvectors_path']).shape[0]

        self.wd = WeightDeformatorFixedBasis(
            generator=self.G,
            conv_layer_ix=self.layer_ix,
            directions_count=directions_count,
            basis_vectors_path=kwargs['eigenvectors_path']
        )

        self.G.style_gan2.convs[self.layer_ix].conv.direction_to_basis_coefs = nn.Parameter(
            torch.eye(directions_count).cuda()
        )

        self.G = self.G.cuda()
        self.directions_count = directions_count

    def __create_wd_hessian_rectification(self, kwargs):
        generator_dict = torch.load(kwargs['checkpoint_path'])['generator']
        dict_key_prefix = f'style_gan2.convs.{self.layer_ix}.conv.'

        direction_to_basis_coefs = generator_dict[dict_key_prefix + 'direction_to_basis_coefs']
        directions_count = direction_to_basis_coefs.shape[0]

        self.wd = WeightDeformatorFixedBasis(
            generator=self.G,
            conv_layer_ix=self.layer_ix,
            directions_count=directions_count,
            basis_vectors_path=kwargs['eigenvectors_path']
        )

        self.G.style_gan2.convs[self.layer_ix].conv.direction_to_basis_coefs = nn.Parameter(
            direction_to_basis_coefs.cuda()
        )

        self.G = self.G.cuda()
        self.directions_count = directions_count


class GeneratorWithFixedWeightDeformation(nn.Module):
    def __init__(self, generator, deformation_path):
        super().__init__()
        self.G = generator

        state_dict = torch.load(deformation_path)
        self.shift_direction = state_dict['shift']
        layer_index = state_dict['layer_ix']
        self.scale = self.shift_direction.norm()

        self.wd = WeightDeformatorFixedBasis(
            generator=self.G,
            conv_layer_ix=layer_index,
            directions_count=1,
            basis_vectors=self.shift_direction / self.scale,
        )

        deformed_conv = self.G.style_gan2.convs[layer_index].conv
        deformed_conv.direction_to_basis_coefs.data = \
            torch.ones_like(deformed_conv.direction_to_basis_coefs.data)

        self.G = self.G.cuda()

    def forward(self, x):
        return self.G(x)

    def deformate(self, batch_shifts):
        self.wd.deformate(0, batch_shifts)
