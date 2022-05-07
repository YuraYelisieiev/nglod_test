import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from lib.trainer import Trainer
from lib.models import OctreeSDF
from lib.options import parse_options

if __name__ == "__main__":
    # TODO: For every feature save model in a folder and check sizes
    # TODO: Find corresponding model
    pre_trained_path = "_results/models/armadillo.pth"
    models_path = '/'.join(pre_trained_path.split('/')[:-1])
    parser = parse_options(return_parser=True)
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    net = globals()['OctreeSDF'](args)

    # For faster inference
    if args.jit:
        net = torch.jit.script(net)

    net.load_state_dict(torch.load(pre_trained_path))

    net.to(device)
    net.eval()

    # Exctract features and for all level of detail
    lods = net.features
    decoders = net.louts
    # Here we are using number of lods to save every LOD model separately
    for lod_n in range(args.num_lods):
        print(f'{lod_n} Done!')
        net.features = nn.ModuleList([lods[lod_n]])
        net.louts = nn.ModuleList([decoders[lod_n]])
        # Change number of lods for each model
        net.num_lods = 1
        torch.save(net.state_dict(), os.path.join(models_path, f"test_lod_{lod_n}.pth"))