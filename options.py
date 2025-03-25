import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=60, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

parser.add_argument('--rgb_root', type=str, default='.../SOD_data/RGBD_for_train/RGB/',
                    help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='.../SOD_data/RGBD_for_train/depth/',
                    help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='.../SOD_data/RGBD_for_train/GT/',
                    help='the training gt images root')

parser.add_argument('--test_rgb_root', type=str, default='.../SOD_data/test_in_train/RGB/',
                    help='the test rgb images root')
parser.add_argument('--test_depth_root', type=str, default='.../SOD_data/test_in_train/depth/',
                    help='the test depth images root')
parser.add_argument('--test_gt_root', type=str, default='.../SOD_data/test_in_train/GT/',
                    help='the test gt images root')

parser.add_argument('--save_path', type=str, default='', help='the path to save models and logs')

parser.add_argument('--pre', type=str, default='.../MambaSOD/models/VMamba/vssm1_tiny_0230s_ckpt_epoch_264.pth', help='pre')
parser.add_argument('--cfg', type=str, default='.../MambaSOD/models/VMamba/vmambav2v_tiny_224.yaml', help='cfg')
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs='+',)
opt = parser.parse_args()
