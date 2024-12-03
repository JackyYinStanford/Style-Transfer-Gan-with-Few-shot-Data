import blend_models
import os
import argparse

parser = argparse.ArgumentParser(description='params to run run_blending.py')

parser.add_argument('--name', type=str, default='baby_600', help='the name of the running instance')
parser.add_argument('--count', type=str, default='000024', help='specify which epoch to blend the models')
parser.add_argument('--prefix', type=str, default='00024', help='specify the prefix of the models')
parser.add_argument('--swap', action='store_true',
                    help='if swap, using the specified swap strategy')
parser.add_argument('--layers', type=int, nargs='+', default=[4, 8, 16, 32, 64, 128, 256, 512, 1024],
                    help='specify which layers set to be used for latter models')
parser.add_argument('--reverse', action='store_true',
                    help='if reversed, using the stylegan2-ffhq model as the low-level models')
parser.add_argument('--once', action='store_true',
                    help='if specified only blend once, then only blend the specified model for once')
args = parser.parse_args()

name = args.name
count = args.count
prefix = args.prefix

resolutions = [16, 32, 64, 128, 256, 512, 1024]

char = '_'
layers = [str(x) for x in args.layers]
swap_layers = char.join(layers)

if args.once:
    vis_path = f'./vis/{name}/{count}/'
    model_path = f'./checkpoints/{name}/{count}/'

    os.makedirs(vis_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    res = 4
    suffix_name = '_'.join(str(i) for i in args.layers)
    filename = f"blended-{suffix_name}.jpg"
    blend_models.main("weights/stylegan2-ffhq-config-f.pkl",
                      "results/00062-stylegan2-thai_male_v1-2gpu-config-f/network-snapshot-000042.pkl",
                      res, output_grid=os.path.join(vis_path, filename),
                      output_pkl=os.path.join(model_path, f"blended-{suffix_name}.pkl"), verbose=True,
                      smooth=args.layers)

if args.swap:
    if args.reverse:
        vis_path = f'./vis/{name}_{swap_layers}_reverse/{count}/'
        model_path = f'./checkpoints/{name}_{swap_layers}_reverse/{count}/'
    else:
        vis_path = f'./vis/{name}_{swap_layers}/{count}/'
        model_path = f'./checkpoints/{name}_{swap_layers}/{count}/'
else:
    if args.reverse:
        vis_path = f'./vis/{name}/{count}_reverse/'
        model_path = f'./checkpoints/{name}/{count}_reverse/'
    else:
        vis_path = f'./vis/{name}/{count}/'
        model_path = f'./checkpoints/{name}/{count}/'

os.makedirs(vis_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

if args.swap:
    filename = f"blended-{swap_layers}.jpg"
    if args.reverse:
        blend_models.main("weights/stylegan2-ffhq-config-f.pkl",
                          f"results/{prefix}-stylegan2-{name}-2gpu-config-f/network-snapshot-{count}.pkl",
                          resolution=4, output_grid=os.path.join(vis_path, filename),
                          output_pkl=os.path.join(model_path, f'blended-{swap_layers}.pkl'),
                          smooth=args.layers)
    else:
        blend_models.main(f"results/{prefix}-stylegan2-{name}-2gpu-config-f/network-snapshot-{count}.pkl",
                          "weights/stylegan2-ffhq-config-f.pkl", resolution=4,
                          output_grid=os.path.join(vis_path, filename),
                          output_pkl=os.path.join(model_path, f'blended-{swap_layers}.pkl'),
                          verbose=True, smooth=args.layers)

# 不使用swap strategy的话则每层都融合一次
else:
    for res in resolutions:
        filename = f"blended-{res}.jpg"
        if os.path.isfile(os.path.join(model_path, f'blended-{res}.pkl')):
            continue
        if args.reverse:
            blend_models.main("weights/stylegan2-ffhq-config-f.pkl",
                              f"results/{prefix}-stylegan2-{name}-2gpu-config-f/network-snapshot-{count}.pkl",
                              res, output_grid=os.path.join(vis_path, filename),
                              output_pkl=os.path.join(model_path, f'blended-{res}.pkl'))
        else:
            blend_models.main(f"results/{prefix}-stylegan2-{name}-2gpu-config-f/network-snapshot-{count}.pkl",
                              "weights/stylegan2-ffhq-config-f.pkl", res,
                              output_grid=os.path.join(vis_path, filename),
                              output_pkl=os.path.join(model_path, f'blended-{res}.pkl'), verbose=True)
