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
from sklearn.metrics import f1_score


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
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            pts = batch[0].cuda()
            gt = batch[1].cuda()


            pred = net(pts)
            gt = sdf_to_classes(gt)
            pred = sdf_to_classes(pred)

            gt = gt.detach().cpu()
            pred = pred.detach().cpu()
            metrics_dict['F1'] += [f1_score(gt, pred, average='micro')]

    return sum(metrics_dict['F1']) / len(metrics_dict['F1'])



if __name__ == "__main__":
    parser = parse_options(return_parser=True)
    app_group = parser.add_argument_group('app')
    app_group.add_argument('--l2-loss', type=float, default=1.0,
                           help='Weight of standard L2 loss')
    app_group.add_argument('--normalize-mesh', action='store_true',
                           help='Normalize the mesh')
    app_group.add_argument('--models-dir', type=str,
                           help='Path to trained models!')
    app_group.add_argument('--meshes-dir', type=str,
                           help='Normalize the mesh')

    app_group.add_argument('--feature-std', type=float, default=0.01,
                           help='Feature initialization distribution')

    args = parser.parse_args()

    # Sufficient enough large value for metric estimation
    num_samples = 100_000
    check_points = os.listdir(args.models_dir)
    print(check_points)

    trained_models = list(
        map(lambda x: os.path.join(args.models_dir, x), check_points)
    )

    g_f1_occ = 0
    g_f1_bv = 0
    args.pretrained = True
    args.sparse_f_n = 3851

    net = globals()['NeuralSPC'](args)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    number_of_samples = len(trained_models)
    for model_ckpt in tqdm(trained_models):
        pre_trained_path = model_ckpt
        models_path = '/'.join(pre_trained_path.split('/')[:-1])
        obj_fn = os.path.basename(pre_trained_path).split('.')[0] + '.obj'
        args.mesh_path = os.path.join(args.meshes_dir, obj_fn)
        args.dataset_path = args.mesh_path

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

        print("Resampling dataset and evaluating metrics!")
        test_dataloader = resamples_dataloader(test_dataset_1, 'near', num_samples)
        f1_occ = calculate_F1(test_dataloader, net)
        g_f1_occ += f1_occ
        print('F1 occupancy near surface:')
        print(f1_occ)

        # New dataset sampled with different mode
        test_dataset_2 = MeshDataset(net.V, net.F, args, num_samples=num_samples)
        test_dataloader_2 = resamples_dataloader(test_dataset_2, 'rand', num_samples)
        f1_bv = calculate_F1(test_dataloader_2, net)
        g_f1_bv += f1_bv

        print('F1 in bounding volume:')
        print(f1_bv)
        del test_dataset_1, test_dataset_2, net, test_dataloader_2, test_dataloader
        torch.cuda.empty_cache()

    print(f"Mean F1 occupancy near surface: {g_f1_occ / number_of_samples}")
    print(f"Mean F1 in bounding volume: {g_f1_bv / number_of_samples}")
