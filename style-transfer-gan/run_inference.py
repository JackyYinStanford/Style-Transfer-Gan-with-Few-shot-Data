import pretrained_networks
import numpy as np
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib
from pathlib import Path
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='params to run run_inference.py')

parser.add_argument('--name', type=str, default='baby_600', help='the name of the running instance')
parser.add_argument('--reverse', action='store_true',
                    help='if reversed, using the stylegan2-ffhq model as the low-level models')
parser.add_argument('--counts', type=str, nargs='+', default=['000024'], help='specify which epoch to blend the models')
parser.add_argument('--prefix', type=str, default='00024', help='specify the prefix of the models')
parser.add_argument('--dataset', type=str, default='test_face_1000',
                    help='specify which reconstructed npy dataset to use')
parser.add_argument('--resolutions', type=str, nargs='+',
                    default=['4', '8', '16', '32', '64', '128', '256', '512', '1024'],
                    help='specify which blended models to be tested on inference stage')
args = parser.parse_args()

name = args.name
counts = args.counts
pre = args.prefix
data_name = args.dataset

resolutions = [16, 32, 64, 128, 256, 512, 1024]
truncation_psi = 0.5

latent_dir = Path(f"generated_dataset/{data_name}")
for resolution in resolutions:
    for count in counts:
        if args.reverse:
            blended_url = './checkpoints/{}/{}_reverse/blended-{}.pkl'.format(name, count, resolution)
            save_path = f'./inference_results/{data_name}/{count}_{resolution}_{truncation_psi}_reverse'
        else:
            blended_url = './checkpoints/{}/{}/blended-{}.pkl'.format(name, count, resolution)
            save_path = f'./inference_results/{data_name}/{name}/{count}_{resolution}_{truncation_psi}'
        os.makedirs(save_path, exist_ok=True)
        ffhq_url = './weights/stylegan2-ffhq-config-f.pkl'

        # 这两个是用来看ffhq/finetune模型在真实人脸上的效果的
        if resolution == 'finetune':
            blended_url = f"results/{pre}-stylegan2-{name}-2gpu-config-f/network-snapshot-{count}.pkl"
        if resolution == 'ori':
            blended_url = './weights/stylegan2-ffhq-config-f.pkl'

        _, _, Gs_blended = pretrained_networks.load_networks(blended_url)
        _, _, Gs = pretrained_networks.load_networks(ffhq_url)
        w_avg = Gs.get_var('dlatent_avg')
        latents = latent_dir.glob("*.npy")
        for latent_file in tqdm(latents):
            if os.path.isfile(os.path.join(save_path, f"{latent_file.stem}.jpg")):
                continue
            latent = np.load(latent_file)
            latent = np.expand_dims(latent, axis=0)
            if truncation_psi < 1:
                latent = w_avg + (latent - w_avg) * truncation_psi
            synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False),
                                    minibatch_size=8)
            images = Gs_blended.components.synthesis.run(latent, randomize_noise=False, **synthesis_kwargs)
            Image.fromarray(images.transpose((0, 2, 3, 1))[0], 'RGB').save(
                os.path.join(save_path, f"{latent_file.stem}.jpg"))
