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
from sklearn.metrics import f1_score, precision_score, recall_score

def sdf_to_classes(sdf):
    sdf[sdf < 0] = 0
    sdf[sdf > 0] = 1
    return sdf


def resamples_dataloader(dataset, mode, num_samples):
    # Creates new dataloader with specific sampling mode
    dataset.resample_sm([mode])
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False, pin_memory=True, num_workers=4)
    return dataloader


def calculate_F1(dataloader, net):
    metrics_dict = dict()
    metrics_dict['F1'] = []
    for idx, batch in enumerate(dataloader):
        pts = batch[0].to(device)
        gt = batch[1].to(device)

        pred = net(pts)
        gt = sdf_to_classes(gt)
        pred = sdf_to_classes(pred)

        gt = gt.detach().cpu()
        pred = pred.detach().cpu()

        print(f'Precision: {precision_score(gt, pred)}')
        print(f'Recall: {recall_score(gt, pred)}')
        metrics_dict['F1'] += [f1_score(gt, pred)]

    print(len(metrics_dict))
    return sum(metrics_dict['F1']) / len(metrics_dict['F1'])

# ['47984.pth', '68380.pth', '79241.pth', '398259.pth', '73075.pth', '53159.pth', '72960.pth', '44234.pth', '64444.pth', '64764.pth', '68381.pth']

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
    app_group.add_argument('--pretrained-path', type=str,
                           help='Normalize the mesh')
    app_group.add_argument('--feature-std', type=float, default=0.01,
                           help='Feature initialization distribution')

    args = parser.parse_args()
    args.dataset_path = args.mesh_path
    # Sufficient enough large value for metric estimation
    num_samples = 137_000
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

    with torch.no_grad():
        test_dataset_1 = MeshDataset(net.V, net.F, args, num_samples=num_samples)
        print("Resampling dataset and evaluating metrics!")
        test_dataloader = resamples_dataloader(test_dataset_1, 'near', num_samples)
        f1_occ = calculate_F1(test_dataloader, net)

        print('F1 occupancy near surface:')
        print(f1_occ)

        # New dataset sampled with different mode
        test_dataloader_2 = resamples_dataloader(test_dataset_1, 'rand', num_samples)
        f1_bb = calculate_F1(test_dataloader_2, net)
        print('F1 in bounding volume:')
        print(f1_bb)
