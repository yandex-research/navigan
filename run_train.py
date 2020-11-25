from constants import DEFORMATOR_TYPE_DICT, DEFORMATOR_TARGET_DICT
from constants import SHIFT_DISTRIDUTION_DICT, WEIGHTS
from loading import load_generator
from latent_deformator import LatentDeformator
from weight_deformator import WeightDeformatorFixedBasis
from weight_deformator import WeightDeformatorSVDBasis
from latent_shift_predictor import LatentShiftPredictor, LeNetShiftPredictor
from trainer import Trainer, Params
from visualization import inspect_all_directions
from utils import make_noise, save_command_run_params
from videos import make_video
from tqdm.auto import trange

import os
import sys
import argparse
import random
import torch

import matplotlib
matplotlib.use("Agg")


def generate_videos(args, generator, deformator):
    clips_dir = os.path.join(args.out, 'videos_rectification')
    if not os.path.isdir(clips_dir):
        os.mkdir(clips_dir)

    if args.samples_for_videos is None:
        dim_z = generator.dim_z
        zs = torch.randn((4, dim_z)).cuda()
    else:
        zs = torch.load(args.samples_for_videos).cuda()

    print('Making videos...')
    for i in trange(args.directions_count):

        def wd_deformate_arguments_builder(shift_):
            directions = torch.LongTensor([i] * len(zs)).cuda()
            shifts = torch.FloatTensor([shift_] * len(zs)).cuda()
            return directions, shifts

        for amplitude in [args.shift_scale, 1.5*args.shift_scale, 2*args.shift_scale]:
            clip_path = os.path.join(clips_dir, f'direction{i}_amplitude{amplitude}')

            make_video(
                generator=generator,
                zs=zs,
                wd=deformator,
                file_dest=clip_path,
                shift_from=-amplitude,
                shift_to=amplitude,
                step=amplitude / 50.,
                interpolate=args.video_interpolate,
                wd_deformate_arguments_builder=wd_deformate_arguments_builder
            )


def main():
    parser = argparse.ArgumentParser(description='Latent space rectification')
    for key, val in Params().__dict__.items():
        target_type = type(val) if val is not None else int
        parser.add_argument('--{}'.format(key), type=target_type, default=None)

    parser.add_argument('--out', type=str, required=True, help='results directory')
    parser.add_argument('--gan_type', type=str, choices=WEIGHTS.keys(), help='generator model type')
    parser.add_argument('--gan_weights', type=str, default=None, help='path to generator weights')
    parser.add_argument('--resolution', type=int, required=True)
    parser.add_argument('--target_class', nargs='+', type=int, default=[239],
                        help='classes to use for conditional GANs')

    parser.add_argument('--deformator', type=str, default='ortho',
                        choices=DEFORMATOR_TYPE_DICT.keys(), help='deformator type')
    parser.add_argument('--deformator_random_init', type=bool, default=True)
    parser.add_argument('--deformator_target', type=str, default='latent',
                        choices=DEFORMATOR_TARGET_DICT.keys())
    parser.add_argument('--deformator_conv_layer_index', type=int, default=3)
    parser.add_argument('--basis_vectors_path', type=str)

    parser.add_argument('--shift_predictor_size', type=int, help='reconstructor resolution')
    parser.add_argument('--shift_predictor', type=str,
                        choices=['ResNet', 'LeNet'], default='ResNet', help='reconstructor type')
    parser.add_argument('--shift_distribution_key', type=str,
                        choices=SHIFT_DISTRIDUTION_DICT.keys())

    parser.add_argument('--make_videos', type=bool, default=True)
    parser.add_argument('--samples_for_videos', type=str, default=None)
    parser.add_argument('--video_interpolate', type=int, default=None)

    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--multi_gpu', type=bool, default=False,
                        help='Run generator in parallel. Be aware of old pytorch versions:\
                              https://github.com/pytorch/pytorch/issues/17345')
    # model-specific
    parser.add_argument('--w_shift', type=bool, default=True,
                        help='latent directions search in w-space for StyleGAN')

    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    save_command_run_params(args)

    # init models
    if args.gan_weights is not None:
        weights_path = args.gan_weights
    else:
        weights_path = WEIGHTS[args.gan_type]

    G = load_generator(args.__dict__, weights_path, args.w_shift)

    if args.deformator_target == 'latent':
        deformator = LatentDeformator(
            shift_dim=G.dim_shift,
            input_dim=args.directions_count,
            out_dim=args.max_latent_dim,
            type=DEFORMATOR_TYPE_DICT[args.deformator],
            random_init=args.deformator_random_init
        ).cuda()
    elif args.deformator_target == 'weight_svd':
        deformator = WeightDeformatorSVDBasis(
            generator=G,
            conv_layer_ix=args.deformator_conv_layer_index,
            directions_count=args.directions_count
        ).cuda()
        G = G.cuda()
        dim_shift = args.directions_count
    elif args.deformator_target == 'weight_fixedbasis':
        assert args.basis_vectors_path is not None
        deformator = WeightDeformatorFixedBasis(
            generator=G,
            conv_layer_ix=args.deformator_conv_layer_index,
            directions_count=args.directions_count,
            basis_vectors_path=args.basis_vectors_path
        ).cuda()
        G = G.cuda()
        dim_shift = args.directions_count
    else:
        raise ValueError("Unknown deformator_target")

    if args.shift_predictor == 'ResNet':
        shift_predictor = LatentShiftPredictor(
            dim_shift, args.shift_predictor_size).cuda()
    elif args.shift_predictor == 'LeNet':
        shift_predictor = LeNetShiftPredictor(
            dim_shift, 1 if args.gan_type == 'SN_MNIST' else 3).cuda()

    # training
    args.shift_distribution = SHIFT_DISTRIDUTION_DICT[args.shift_distribution_key]

    params = Params(**args.__dict__)
    # update dims with respect to the deformator if some of params are None
    if args.deformator_target == 'latent':
        params.directions_count = int(deformator.input_dim)
        params.max_latent_dim = int(deformator.out_dim)

    trainer = Trainer(params, out_dir=args.out)
    trainer.train(G, deformator, shift_predictor, multi_gpu=args.multi_gpu)

    if args.make_videos:
        if 'weight_' not in args.deformator_target:
            sys.stderr.write("Making video is available only for weight deformations.\n")
        else:
            generate_videos(args, G, deformator)

    if args.deformator_target == 'latent':
        save_results_charts(G, deformator, params, trainer.log_dir)


def save_results_charts(G, deformator, params, out_dir):
    deformator.eval()
    G.eval()
    z = make_noise(3, G.dim_z, params.truncation).cuda()
    inspect_all_directions(
        G, deformator, os.path.join(out_dir, 'charts_s{}'.format(int(params.shift_scale))),
        zs=z, shifts_r=params.shift_scale)
    inspect_all_directions(
        G, deformator, os.path.join(out_dir, 'charts_s{}'.format(int(3 * params.shift_scale))),
        zs=z, shifts_r=3 * params.shift_scale)


if __name__ == '__main__':
    main()
