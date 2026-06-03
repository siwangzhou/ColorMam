import argparse
import os
from util import util
import torch


class ExOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
#         self.parser.add_argument('--dataroot', required=False, default='/root/data/vjuicefs_ai_camera_jgroup_livephoto/11182563/color_transfer/data_crop_4_lut', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--dataroot', required=False, default='/media/good-person/新加卷/code/Python/project/mechine_Learning/Data/colorTransfer', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')

        self.parser.add_argument('--val_dataroot', required=False, default='/root/data/vjuicefs_ai_camera_jgroup_livephoto/11182563/color_transfer/val_data/', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')

        self.parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')

        self.parser.add_argument('--checkpoints_dir', type=str, default='./weight/manba', help='models are saved here')
        self.parser.add_argument('--network', type=str, default='iccv_submitted')
        self.parser.add_argument('--network_H', type=str, default='basic')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_env', type=str, default='main', help='Environment name of the web display')
        self.parser.add_argument('--display_port', type=int, default=6005, help='visdom port of the web display')
        self.parser.add_argument('--no_dropout', action='store_false', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--img_type', type=str, default='lab', help='Environment name of the web display')
        self.parser.add_argument('--pair_ratio', type=float, default = 0.0, help='Ratio of Pair data')
        self.parser.add_argument('--mode', type=str, default='gsgt', help='gsgt, gsrt, rsrt')
        self.parser.add_argument('--test_dir', type=str, default='1', help='1,2,3,4,5')
        self.parser.add_argument('--is_psnr', action='store_true', help='1,2,3,4,5')
        self.parser.add_argument('--is_SR', action='store_true', help='1,2,3,4,5')
        self.parser.add_argument('--display_freq', type=int, default=1, help='frequency of showing training results on screen')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=20, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--delete_log_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
        
        self.parser.add_argument('--continue_train', type=bool, default=True , help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        # self.parser.add_argument('--ckpt_name', type=str, default=None, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam') #initial 0.0002
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--identity', type=float, default=0.5, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')

        self.parser.add_argument('--shuffle', type=bool, default=True, help='dataloader shuffle')
#         self.parser.add_argument('--ckpt_name', type=str, default='/root/data/vjuicefs_ai_camera_jgroup_livephoto/11182563/color_transfer/methodOmni/weight/v2/epoch_good.pth', help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--ckpt_name', type=str, default='./weight/manba/epoch_274.pth', help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--epochs', type=int, default=800, help='total epochs of training')
        self.parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
        self.parser.add_argument('--display_dir', type=str, default='./results', help='frequency of showing training results on screen')
        self.parser.add_argument('--mamba_from_trion', type=int, default=1, help='frequency of showing training results on screen')
        self.initialized = True
        self.isTrain = True


        self.parser.add_argument('--nVSSMs', type=int, default=2)
        self.parser.add_argument('--nSAVSSMs', type=int, default=2)
        self.parser.add_argument('--nSAVSSGs', type=int, default=2)
        self.parser.add_argument('--embed-dim', type=int, default=64)
        self.parser.add_argument('--patch-size', type=int, default=1)
        self.parser.add_argument('--representation-dim', type=int, default=128)
        self.parser.add_argument('--d-state', type=int, default=16)
        self.parser.add_argument('--expand', type=float, default=2.0)
        self.parser.add_argument('--compress-ratio', type=int, default=8)
        self.parser.add_argument('--squeeze-factor', type=int, default=8)
        self.parser.add_argument('--mamba-from-trion', type=int, default=0)

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        return self.opt
