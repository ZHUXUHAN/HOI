from config import cfg,merge_cfg_from_file, merge_cfg_from_list
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Pet Model Training')
parser.add_argument('--cfg', dest='cfg_file',
                    help='optional config file',
                    default='./cfgs/ssd/voc/ssd_vgg16_300x300_1x.yaml', type=str)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('opts', help='See pet/ssd/core/config.py for all options',
                    default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()

if args.cfg_file is not None:
    merge_cfg_from_file(args.cfg_file)
if args.opts is not None:
    merge_cfg_from_list(args.opts)

print(cfg.TRAIN.WEIGHTS)