import blend_models
import dnnlib.tflib as tflib
import pretrained_networks
import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='params to run blend_inference_multiple.py')

parser.add_argument('--name', type=str, default='baby_600', help='the name of the running instance')

parser.add_argument('--count', nargs='+', type=str, default=['000024'], help='specify which epoch to blend the models')
parser.add_argument('--prefix', type=str, default='00024', help='specify the prefix of the models')
parser.add_argument('--swap', action='store_true',
                    help='if swap, using the specified swap strategy')
parser.add_argument('--layers', type=int, nargs='+', default=[128, 256],
                    help='specify which layers set to be used for latter models')
parser.add_argument('--dataset', type=str, default='test_face',
                    help='specify which reconstructed npy dataset to use')
args = parser.parse_args()

name = args.name
counts = args.count
prefix = args.prefix

# prefix和name要一一对应

for count in counts:
    # 每个count对应创建一个count的文件夹用来放可视化和模型结果
    vis_path = f'./vis/{name}/{count}/'
    model_path = f'./checkpoints/{name}/{count}/'
    os.makedirs(vis_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    # 针对不同的res进行融合看效果，res之后用的是
    for res in args.layers:

        filename = os.path.join(vis_path, f"blended-{res}.jpg")
        checkpoint = os.path.join(model_path, f'blended-{res}.pkl')
        if not os.path.isfile(checkpoint):
            # 大于res层的会用ffhq，小于res层的会用finetune结果，这里没有使用blend_width，默认全为model1或全为model2
            blend_models.main(f"results/{prefix}-stylegan2-{name}-2gpu-config-f/network-snapshot-{count}.pkl",
                              "weights/stylegan2-ffhq-config-f.pkl", res, output_grid=filename,
                              output_pkl=checkpoint, verbose=True)

        truncation_psi = 0.5

        data_name = args.dataset
        latent_dir = Path(f"generated_dataset/{data_name}")

        save_pth = f'./inference_results/{data_name}/{name}/{count}/{res}'
        os.makedirs(save_pth, exist_ok=True)
        blended_url = checkpoint
        ffhq_url = './weights/stylegan2-ffhq-config-f.pkl'
        _, _, Gs_blended = pretrained_networks.load_networks(blended_url)
        _, _, Gs = pretrained_networks.load_networks(ffhq_url)
        w_avg = Gs.get_var('dlatent_avg')
        latents = latent_dir.glob("*.npy")
        for latent_file in tqdm(latents):
            if os.path.isfile(os.path.join(save_pth, f"{latent_file.stem}.jpg")):
                continue
            latent = np.load(latent_file)
            latent = np.expand_dims(latent, axis=0)
            if truncation_psi < 1:
                latent = w_avg + (latent - w_avg) * truncation_psi
            synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False),
                                    minibatch_size=8)
            images = Gs_blended.components.synthesis.run(latent, randomize_noise=False, **synthesis_kwargs)
            Image.fromarray(images.transpose((0, 2, 3, 1))[0], 'RGB').save(
                os.path.join(save_pth, f"{latent_file.stem}.jpg"))
