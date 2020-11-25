import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from weight_deformator import get_conv_from_generator
from copy import deepcopy
from gans import gan_load
import lpips

from videos import make_video


def orthogonal_complement(v, subspace_basis_vectors):
    v = v.detach().clone()

    for basis_vector in subspace_basis_vectors:
        basis_vector = basis_vector.cuda()
        basis_vector_norm = torch.sqrt(basis_vector.pow(2).sum())
        basis_vector /= basis_vector_norm

        dot_product = (v * basis_vector).sum()
        v -= basis_vector * dot_product

    return v


class HVP_forward(torch.nn.Module):
    def __init__(self, generator, lpips_model, lpips_interpolation_size,
                 conv_layer_ix=3, batch_size=8, cache_path='.',
                 update_zs_every_step=False):
        super(HVP_forward, self).__init__()

        self.generator_0 = generator.cuda().eval()
        self.generator_1 = deepcopy(generator).cuda()
        self.lpips_model = lpips_model

        self.conv_layer_0 = [get_conv_from_generator(self.generator_0, conv_layer_ix)]
        self.conv_layer_1 = [get_conv_from_generator(self.generator_1, conv_layer_ix)]

        self.batch_size = batch_size
        self.cache_path = cache_path
        self.lpips_interpolation_size = lpips_interpolation_size
        self.update_zs_every_step = update_zs_every_step

    def calculate_g_batch(self, zs, weight_delta):
        with torch.no_grad():
            img_0 = self.generator_0(zs)
            img_0 = F.interpolate(
                img_0, size=(self.lpips_interpolation_size, self.lpips_interpolation_size))

        self.conv_layer_1[0].weight = nn.Parameter(
            self.conv_layer_0[0].weight.data + weight_delta)

        img_1 = self.generator_1(zs)
        img_1 = F.interpolate(
            img_1, size=(self.lpips_interpolation_size, self.lpips_interpolation_size))

        lpips_distance = self.lpips_model(img_0, img_1).mean()

        self.zero_grad()
        lpips_distance.backward()

        return self.conv_layer_1[0].weight.grad

    def calculate_g(self, zs, weight_delta):
        assert len(zs) % self.batch_size == 0
        batches_cnt = len(zs) // self.batch_size

        g = None
        for ix in range(0, len(zs), self.batch_size):
            z_batch = zs[ix : (ix + self.batch_size)]
            g_batch = self.calculate_g_batch(z_batch, weight_delta)

            if g is None:
                g = g_batch / batches_cnt
            else:
                g += g_batch / batches_cnt
        return g

    def forward_step(self, zs, v, epsilon):
        g_z_plus_delta = self.calculate_g(zs, epsilon * v)  # g(w + epsilon * v)
        g_z_minus_delta = self.calculate_g(zs, -epsilon * v)  # g(w - epsilon * v)
        norm = torch.sqrt(v.pow(2).sum())
        return (g_z_plus_delta - g_z_minus_delta) / (2 * (epsilon + 1e-14) * norm)
   
    def find_eigenvector(self, zs, projector_to_orthogonal_subspace, max_iter, epsilon):
        v_current = torch.randn(self.conv_layer_0[0].weight.data.shape).cuda()
        v_current = projector_to_orthogonal_subspace(v_current)

        for i in range(max_iter):
            v_new = self.forward_step(zs, v_current, epsilon)
            v_new = projector_to_orthogonal_subspace(v_new)

            norm_diff = torch.sqrt((v_new - v_current).pow(2).sum())
            print(f'Step: {i + 1}.\tNorm of (v_new - v_current): {norm_diff}')

            v_current = v_new

        return v_current

    def find_top_n_eigenvectors(self, n=10, num_samples=64, max_iter=20, epsilon=0.1):
        zs = self.load_or_generate_zs(num_samples)

        eigenvectors = self.load_eigenvectors()
        for i in range(len(eigenvectors), n):
            if self.update_zs_every_step:
                zs = torch.randn((num_samples, self.generator_0.dim_z)).cuda()

            print(f'Finding eigenvector #{i + 1}')
            projector = lambda v: orthogonal_complement(v, eigenvectors)

            new_eigenvector = self.find_eigenvector(zs, projector, max_iter, epsilon).cpu().unsqueeze(0)
            if isinstance(eigenvectors, list): # empty list
                eigenvectors = new_eigenvector
            else:
                eigenvectors = torch.cat((eigenvectors, new_eigenvector))
            self.save_eigenvectors(eigenvectors)

        return eigenvectors

    def load_or_generate_zs(self, num_samples):
        zs_path = os.path.join(self.cache_path, 'zs.tmp.pt')
        try:
            zs = torch.load(zs_path).cuda()
            print('Restored cached zs')
            assert len(zs) == num_samples, 'Saved zs has number of points different from num_samples'
        except:
            zs = torch.randn((num_samples, self.generator_0.dim_z)).cuda()
            torch.save(zs, zs_path)
        return zs

    def load_eigenvectors(self):
        eigenvectors_path = os.path.join(self.cache_path, 'eigenvectors.tmp.pt')
        try:
            eigenvectors = torch.load(eigenvectors_path).cpu()
            print(f'Restored {len(eigenvectors)} cached eigenvectors')
        except:
            eigenvectors = list()
        return eigenvectors

    def save_eigenvectors(self, eigenvectors):
        eigenvectors_path = os.path.join(self.cache_path, 'eigenvectors.tmp.pt')
        torch.save(eigenvectors, eigenvectors_path)

    def remove_cache(self):
        zs_path = os.path.join(self.cache_path, 'zs.tmp.pt')
        eigenvectors_path = os.path.join(self.cache_path, 'eigenvectors.tmp.pt')
        for path in [zs_path, eigenvectors_path]:
            if os.path.isfile(path):
                os.remove(path)


class ConstantWeightDeformator(nn.Module):
    def __init__(self, generator, conv_layer_ix, direction):
        super(ConstantWeightDeformator, self).__init__()
        self.generator = generator
        self.conv = [get_conv_from_generator(self.generator, conv_layer_ix)]
        self.direction = direction
        self.original_weight = self.conv[0].weight.data

    def deformate(self, epsilon):
        self.conv[0].weight = nn.Parameter(
            self.original_weight + epsilon * self.direction)

    def disable_deformation(self):
        self.deformate(.0)

    def __del__(self):
        self.disable_deformation()


def generate_videos(args, eigenvectors):
    clips_dir = os.path.join(args.out, 'videos')
    if not os.path.isdir(clips_dir):
        os.mkdir(clips_dir)

    if args.samples_for_videos is None:
        dim_z = gan_load.load_generator(
            gan_load.GANType.StyleGAN2,
            resolution=args.resolution,
            weights=args.gan_weights
        ).dim_z
        zs = torch.randn((4, dim_z)).cuda()
    else:
        zs = torch.load(args.samples_for_videos).cuda()

    print('Making videos...')
    for i, eigenvector in tqdm(enumerate(eigenvectors)):
        eigenvector = eigenvector.cuda()

        for amplitude in [10, 50, 100, 200, 500]:
            generator = gan_load.load_generator(
                gan_load.GANType.StyleGAN2,
                resolution=args.resolution,
                weights=args.gan_weights
            )

            wd = ConstantWeightDeformator(
                generator=generator,
                conv_layer_ix=args.gan_conv_layer_index,
                direction=eigenvector
            )

            clip_path = os.path.join(clips_dir, f'direction{i}_amplitude{amplitude}')

            make_video(
                generator=generator,
                zs=zs,
                wd=wd,
                file_dest=clip_path,
                shift_from=-amplitude,
                shift_to=amplitude,
                step=amplitude / 50.,
                interpolate=args.video_interpolate
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gan_weights', type=str, required=True)
    parser.add_argument('--resolution', type=int, required=True)
    parser.add_argument('--gan_conv_layer_index', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_eigenvectors', type=int, required=True)
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--max_iter', type=int, default=20)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--lpips_net', type=str, default='vgg')
    parser.add_argument('--make_videos', type=bool, default=True)
    parser.add_argument('--samples_for_videos', type=str, default=None)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--lpips_interpolation_size', type=int, default=64)
    parser.add_argument('--video_interpolate', type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.out):
        print(f'Out: {args.out}')
        os.mkdir(args.out)

    lpips_model = lpips.LPIPS(net=args.lpips_net).cuda()

    generator = gan_load.load_generator(
        gan_load.GANType.StyleGAN2,
        resolution=args.resolution,
        weights=args.gan_weights
    )

    hvp = HVP_forward(
        generator=generator,
        lpips_model=lpips_model,
        conv_layer_ix=args.gan_conv_layer_index,
        batch_size=args.batch_size,
        cache_path=args.out,
        lpips_interpolation_size=args.lpips_interpolation_size
    )

    eigenvectors = hvp.find_top_n_eigenvectors(
        n=args.num_eigenvectors,
        num_samples=args.num_samples,
        max_iter=args.max_iter,
        epsilon=args.epsilon
    )

    eigenvectors = torch.stack([
        F.normalize(eig.view(-1), p=2, dim=0).view(eig.shape)
        for eig in eigenvectors
    ], dim=0)

    gan_name = os.path.split(args.gan_weights)[1].split('.')[0]
    save_path = os.path.join(
        args.out, f'eigenvectors_layer{args.gan_conv_layer_index}_{gan_name}.pt')
    torch.save(eigenvectors, save_path)

    hvp.remove_cache()

    if args.make_videos:
        generate_videos(args=args, eigenvectors=eigenvectors)
