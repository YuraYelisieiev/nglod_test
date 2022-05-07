import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import kaolin

kaolin.ops.random.manual_seed(0, 0, 0)
import torch
from tqdm import tqdm
import torch.nn as nn
from lib.trainer import Trainer
from time import perf_counter_ns
from torch.utils.data import Dataset, DataLoader
from lib.datasets import MeshDataset
from lib.models import OctreeSDF
from app.spc.NeuralSPC import NeuralSPC
from lib.options import parse_options

if __name__ == "__main__":
    # TODO: For every feature save model in a folder and check sizes
    # TODO: Find corresponding model

    parser = parse_options(return_parser=True)
    app_group = parser.add_argument_group('app')
    app_group.add_argument('--l2-loss', type=float, default=1.0,
                           help='Weight of standard L2 loss')
    app_group.add_argument('--mesh-path', type=str,
                           help='Path of SPC mesh')
    app_group.add_argument('--normalize-mesh', action='store_true',
                           help='Normalize the mesh')
    app_group.add_argument('--feature-std', type=float, default=0.01,
                           help='Feature initialization distribution')
    app_group.add_argument('--pretrained-path', type=str)

    args = parser.parse_args()
    args.dataset_path = args.mesh_path
    # Sufficient enough large value for metric estimation
    num_samples = 100_000

    pre_trained_path = args.pretrained_path
    models_path = '/'.join(pre_trained_path.split('/')[:-1])

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    features_n = torch.load(pre_trained_path)['features.0'].shape[0]

    args.pretrained = True
    args.sparse_f_n = features_n
    net = globals()['NeuralSPC'](args)

    print("Loading model!")
    net.load_state_dict(torch.load(pre_trained_path))
    net.to(device)
    net.eval()

    test_dataset_1 = MeshDataset(net.V, net.F, args, num_samples=num_samples)
    test_dataloader = DataLoader(test_dataset_1, batch_size=num_samples, shuffle=False, pin_memory=True, num_workers=4)

    print("Test forwarding time consumption eval loop!")
    total_time = 0
    time_per_point = 0
    for idx, batch in enumerate(test_dataloader):
        pts = batch[0].to(device)
        gt = batch[1].to(device)

        batch_start_t = perf_counter_ns()
        net(pts)
        batch_stop_t = perf_counter_ns()

        batch_time = batch_stop_t - batch_start_t
        point_time = batch_time / len(pts)
        total_time += batch_time
        time_per_point += point_time

    point_time /= len(test_dataloader)
    total_time /= len(test_dataloader)
    print(f'Time per batch ms: {total_time * 1e-6}')
    print(f'Time per point ns: {point_time}')
