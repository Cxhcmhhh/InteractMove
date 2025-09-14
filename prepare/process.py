import os, sys
sys.path.append(os.path.abspath('.'))
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='HumanML3D')
args = parser.parse_args()

if args.dataset == 'InteractMove':
    from prepare.datasets.InteractMove.InteractMove import InteractMove as Dataset
else:
    raise NotImplementedError

Dataset(args.data_dir).process()