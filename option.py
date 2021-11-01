import sys
import argparse

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip(): continue
        yield str(arg)

parser = argparse.ArgumentParser(
    description="Option for training, evaluating, and inferring",
    fromfile_prefix_chars="@",
    conflict_handler="resolve",
)

parser.convert_arg_line_to_args = convert_arg_line_to_args

# Data and Directory setting
parser.add_argument("--root", default=".", type=str, help="Root folder to save data in")
parser.add_argument("--resume", default="", type=str, help="Resume from checkpoint")

parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")

parser.add_argument("--dataset", default="nyu", type=str, help="Dataset to train on")
parser.add_argument("--data_path", default="../dataset/nyu/sync/", type=str, help="path to dataset")
parser.add_argument("--gt_path", default="../dataset/nyu/sync/", type=str, help="path to dataset")
parser.add_argument("--filenames_file", default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt", type=str, help="path to the filenames text file")
parser.add_argument("--data_path_eval", default="../dataset/nyu/official_splits/test/", type=str, help="path to the data for online evaluation")
parser.add_argument("--gt_path_eval", default="../dataset/nyu/official_splits/test/", type=str, help="path to the groundtruth data for online evaluation")
parser.add_argument("--filenames_file_eval", default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt", type=str, help="path to the filenames text file for online evaluation",)

parser.add_argument("--input_height", type=int, help="input height", default=416)
parser.add_argument("--input_width", type=int, help="input width", default=544)
parser.add_argument("--max_depth", type=float, help="maximum depth in estimation", default=10)
parser.add_argument("--min_depth", type=float, help="minimum depth in estimation", default=1e-3)
parser.add_argument("--max_depth_eval", type=float, help="maximum depth for evaluation", default=10)
parser.add_argument("--min_depth_eval", type=float, help="minimum depth for evaluation",default=1e-3)

parser.add_argument("--do_random_rotate", default=True, help="if set, will perform random rotation for augmentation", action="store_true")
parser.add_argument("--degree", type=float, help="random rotation maximum degree", default=2.5)
parser.add_argument("--eigen_crop", default=True,help="if set, crops according to Eigen NIPS14", action="store_true",)
parser.add_argument("--garg_crop", help="if set, crops according to Garg  ECCV16", action="store_true")
parser.add_argument("--do_kb_crop", help="if set, crop input images as kitti benchmark images", action="store_true")
parser.add_argument("--use_right", help="if set, will randomly use right images when train on KITTI", action="store_true")

# Training setting
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument("--epochs", default=25, type=int, help="number of total epochs to run")
parser.add_argument("--bs", default=16, type=int, help="batch size")
parser.add_argument("--max_momentum", type=float, help="maximum depth for evaluation", default=0.95)
parser.add_argument("--wd", "--weight-decay", default=0.1, type=float, help="weight decay")
parser.add_argument("--controller-input", "--controller_input", default="io", type=str, help="Input for controller module: i, o, io")
#parser.add_argument('--variance_focus', type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

#Evaluation and Inference setting
parser.add_argument('--save-dir', '--save_dir', default=None, type=str, help='Store predictions in folder')
parser.add_argument('--checkpoint-path', '--checkpoint_path', type=str, required=False, help="checkpoint file to use for prediction")
parser.add_argument("--baseline", help="if set, ignores controller and use unweighted average to combine predictions instead", action="store_true")

parser.add_argument("--lr", "--learning-rate", default=0.000357, type=float, help="max learning rate")
parser.add_argument("--same-lr", "--same_lr", default=False, action="store_true", help="Use same LR for all param groups")
parser.add_argument("--div-factor", "--div_factor", default=25, type=float, help="Initial div factor for lr")
parser.add_argument("--final-div-factor", "--final_div_factor", default=100, type=float, help="final div factor for lr")

parser.add_argument("--validate-every", "--validate_every", default=100, type=int, help="validation period")

# Model setting
parser.add_argument("--module", default="controller", type=str, help="Module to operate on")
parser.add_argument("--encoder", default="ghostnet_1x", type=str, help="encoder architectures")
parser.add_argument("--encoder-controller", "--encoder_controller", default="ghostnet_1x", type=str, help="encoder architectures")
parser.add_argument("--n-bins", "--n_bins", default=256, type=int, help="number of bins/buckets to divide depth range into")
parser.add_argument("--w_chamfer", "--w-chamfer", default=0.1, type=float, help="weight value for chamfer loss")
parser.add_argument("--bts_size", "--bts_size", default=512, type=int, help="number of bts class")
parser.add_argument("--norm", default="linear", type=str, help="Type of norm/competition for bin-widths", choices=["linear", "softmax", "sigmoid"])

# GPU parallel process setting
parser.add_argument("--gpu", default=None, type=int, help="Which gpu to use")
parser.add_argument("--distributed", default=True, action="store_true", help="Use DDP if set")

# Wandb Logging
parser.add_argument("--notes", default="", type=str, help="Wandb notes")
parser.add_argument("--tags", default="sweep", type=str, help="Wandb tags")

# For finding optimal lr
parser.add_argument("--lrtest", default=50, type=int, help="number of step for lr test")

# For inspect network architectures (summary.py)
parser.add_argument("--inspect-module", "--inspect_module", default="ensemble", type=str, help="Module to inspect on: adabins, bts, ldrn, controller, ensemble")

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = "@" + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else: args = parser.parse_args()

