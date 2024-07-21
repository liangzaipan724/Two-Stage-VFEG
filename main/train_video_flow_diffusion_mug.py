import argparse
import torch.multiprocessing as mp
import imageio
import torch
from torch.utils import data
import numpy as np
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import timeit
import math
from PIL import Image
from misc import Logger, grid2fig, conf2fig
# from datasets_mead import MEAD
from datasets_mug import MUG
import sys
import random
from modules.video_flow_diffusion_model import FlowDiffusion
from torch.optim.lr_scheduler import MultiStepLR
os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
start = timeit.default_timer()
BATCH_SIZE = 6
MAX_EPOCH = 1200
epoch_milestones = [800, 1000]  #用于指定学习率调度器(learning rate scheduler)在训练过程中何时减小学习率。
root_dir = '/data/main/'
data_dir = "/data/mug/"
GPU = "7"
postfix = "-j-sl-vr-of-tr-rmm"
joint = "joint" in postfix or "-j" in postfix  # allow joint training with unconditional model          true
if "random" in postfix:
    frame_sampling = "random"
elif "-vr" in postfix:
    frame_sampling = "very_random"          #very_random抽样
else:
    frame_sampling = "uniform"
only_use_flow = "onlyflow" in postfix or "-of" in postfix  # whether only use flow loss      只使用光流loss
if joint:
    null_cond_prob = 0.1           #null_cond_prob 可能表示某种条件下的空值概率
else:
    null_cond_prob = 0.0
split_train_test = "train" in postfix or "-tr" in postfix
use_residual_flow = "-rf" in postfix
config_pth = "/data/config/config.yaml"
AE_RESTORE_FROM = "/data/main/MUG.pth"
INPUT_SIZE = 128
N_FRAMES = 40
LEARNING_RATE = 2e-4
RANDOM_SEED = 1234
MEAN = (0.0, 0.0, 0.0)
RESTORE_FROM = ""
local_rank=[1,2]
SNAPSHOT_DIR = os.path.join(root_dir, 'snapshots'+postfix)
IMGSHOT_DIR = os.path.join('imgs'+postfix)
VIDSHOT_DIR = os.path.join(root_dir, "vidshots"+postfix)
SAMPLE_DIR = os.path.join(root_dir, 'sample'+postfix)
NUM_EXAMPLES_PER_EPOCH = 465                                 #每个epoch抽样多少
NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))
MAX_ITER = max(NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + 1,
               NUM_STEPS_PER_EPOCH * BATCH_SIZE * MAX_EPOCH + 1)
SAVE_MODEL_EVERY = NUM_STEPS_PER_EPOCH * (MAX_EPOCH // 4)
SAVE_VID_EVERY = 1000
SAMPLE_VID_EVERY = 2000
UPDATE_MODEL_EVERY = 3000

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(IMGSHOT_DIR, exist_ok=True)
os.makedirs(VIDSHOT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

LOG_PATH = SNAPSHOT_DIR + "/B"+format(BATCH_SIZE, "04d")+"E"+format(MAX_EPOCH, "04d")+".log"
sys.stdout = Logger(LOG_PATH, sys.stdout)           #写日志

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Flow Diffusion")
    parser.add_argument("--fine-tune", default=False)
    parser.add_argument("--set-start", default=False)
    parser.add_argument("--start-step", default=0, type=int)
    parser.add_argument("--img-dir", type=str, default=IMGSHOT_DIR,
                        help="Where to save images of the model.")
    parser.add_argument("--local_rank",default=[0,1],help="local device id on current node",type=int)
    parser.add_argument("--num-workers", default=8)
    parser.add_argument("--final-step", type=int, default=int(NUM_STEPS_PER_EPOCH * MAX_EPOCH),
                        help="Number of training steps.")
    parser.add_argument("--gpu", default=GPU,
                        help="choose gpu device.")
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--save-img-freq', default=2, type=int,
                        metavar='N', help='save image frequency')
    parser.add_argument('--save-vid-freq', default=SAVE_VID_EVERY, type=int)
    parser.add_argument('--sample-vid-freq', default=SAMPLE_VID_EVERY, type=int)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--n-frames", default=N_FRAMES)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", default=RESTORE_FROM)
    parser.add_argument("--save-pred-every", type=int, default=SAVE_MODEL_EVERY,
                        help="Save checkpoint every often.")
    parser.add_argument("--update-pred-every", type=int, default=UPDATE_MODEL_EVERY)
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--fp16", default=False)
    return parser.parse_args()


args = get_arguments()


def sample_img(rec_img_batch, idx=0):               #这个函数的作用是从tensor中采样和预处理一张图像,隐藏了tensor到numpy的转换和预处理细节
    rec_img = rec_img_batch[idx].permute(1, 2, 0).data.cpu().numpy().copy()
    rec_img += (np.array(MEAN)/255.0).astype(np.uint8)
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)


def main():
    """Create the model and start the training."""

    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(args.random_seed)
    model = FlowDiffusion(lr=LEARNING_RATE,
                          is_train=True,
                          img_size=INPUT_SIZE//4,
                          num_frames=N_FRAMES,
                          null_cond_prob=null_cond_prob,
                          sampling_timesteps=1000,
                          only_use_flow=only_use_flow,
                          use_residual_flow=use_residual_flow,
                          config_pth=config_pth,
                          pretrained_pth=AE_RESTORE_FROM)
    model.cuda()
    if args.fine_tune:
        pass
    elif args.restore_from:
        if os.path.isfile(args.restore_from):
            print("=> loading checkpoint '{}'".format(args.restore_from))
            checkpoint = torch.load(args.restore_from)
            if args.set_start:
                args.start_step = int(math.ceil(checkpoint['example'] / args.batch_size))
            model_ckpt = model.diffusion.state_dict()
            print('model_ckpt',model_ckpt)
            for name, _ in model_ckpt.items():
                model_ckpt[name].copy_(checkpoint['diffusion'][name])
            model.diffusion.load_state_dict(model_ckpt)
            print("=> loaded checkpoint '{}'".format(args.restore_from))
            if "optimizer_diff" in list(checkpoint.keys()):
                model.optimizer_diff.load_state_dict(checkpoint['optimizer_diff'])
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_from))
    else:
        print("NO checkpoint found!")

    setup_seed(args.random_seed)
    trainloader = data.DataLoader(MUG(data_dir=data_dir,
                                       image_size=INPUT_SIZE,
                                       num_frames=N_FRAMES,
                                       color_jitter=True,
                                       sampling=frame_sampling,
                                       mean=MEAN),
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_rec = AverageMeter()
    losses_warp = AverageMeter()

    cnt = 0
    actual_step = args.start_step
    start_epoch = int(math.ceil((args.start_step * args.batch_size)/NUM_EXAMPLES_PER_EPOCH))
    epoch_cnt = start_epoch
    # optimizer=model.module.optimizer_diff
    scheduler = MultiStepLR(model.optimizer_diff, epoch_milestones, gamma=0.1, last_epoch=start_epoch - 1)
    print("epoch %d, lr= %.7f" % (epoch_cnt, model.optimizer_diff.param_groups[0]["lr"]))

    while actual_step < args.final_step:   #final_step=  max_epoch✖️465(每个epoch多少步）
        iter_end = timeit.default_timer()     #获取当前时间

        for i_iter, batch in enumerate(trainloader):
            actual_step = int(args.start_step + cnt)
            print(actual_step)
            data_time.update(timeit.default_timer() - iter_end)

            real_vids, ref_texts, real_names = batch
            ref_imgs = real_vids[:, :, 0, :, :].clone().detach()

            bs = real_vids.size(0)


            model.set_train_input(ref_img=ref_imgs, real_vid=real_vids, ref_text=ref_texts)
            model.optimize_parameters()

            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()

            losses.update(model.loss, bs)
            losses_rec.update(model.rec_loss, bs)
            losses_warp.update(model.rec_warp_loss, bs)

            if actual_step % args.print_freq == 0:
                print('iter: [{0}]{1}/{2}\t'
                      'loss {loss.val:.7f} ({loss.avg:.7f})\t'
                      'loss_rec {loss_rec.val:.4f} ({loss_rec.avg:.4f})\t'
                      'loss_warp {loss_warp.val:.4f} ({loss_warp.avg:.4f})'
                    .format(
                    cnt, actual_step, args.final_step,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    loss_rec=losses_rec,
                    loss_warp=losses_warp,
                ))

            null_cond_mask = np.array(model.diffusion.denoise_fn.null_cond_mask.data.cpu().numpy(),
                                      dtype=np.uint8)

            if actual_step % args.save_img_freq == 0:
                # print(ref_imgs)
                msk_size = ref_imgs.shape[-1]
                save_src_img = sample_img(ref_imgs)
                save_tar_img = sample_img(real_vids[:, :, N_FRAMES//2, :, :])         #从真实视频序列real_vids中取第N_FRAMES/2帧(中间帧),
                save_real_out_img = sample_img(model.real_out_vid[:, :, N_FRAMES//2, :, :])

                new_im = Image.new('RGB', (msk_size * 5, msk_size * 2))
                new_im.paste(Image.fromarray(save_src_img, 'RGB'), (0, 0))
                new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, msk_size))
                new_im.paste(Image.fromarray(save_real_out_img, 'RGB'), (msk_size, 0))
                new_im_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") \
                              + '_' + real_names[0] + "_%d.png" % (null_cond_mask[0])
                new_im_file = os.path.join(args.img_dir, new_im_name)
                new_im.save(new_im_file)

            if actual_step % args.save_vid_freq == 0 and cnt != 0:
                print("saving video...")
                new_im_arr_list = []
                new_vid_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") \
                                + '_' + real_names[0] + "_%d.gif" % (null_cond_mask[0])
                new_vid_file = os.path.join(VIDSHOT_DIR, new_vid_name)
                imageio.mimsave(new_vid_file, new_im_arr_list)

            if actual_step % args.save_pred_every == 0 and cnt != 0:
                print('taking snapshot ...')
                torch.save({'example': actual_step * args.batch_size,
                            'diffusion': model.diffusion.state_dict(),
                            'optimizer_diff': model.optimizer_diff.state_dict()},
                           osp.join(args.snapshot_dir,
                                    'flowdiff_' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))

            # update saved model
            if actual_step % args.update_pred_every == 0 and cnt != 0:
                print('updating saved snapshot ...')
                torch.save({'example': actual_step * args.batch_size,
                            'diffusion': model.diffusion.state_dict(),
                            'optimizer_diff': model.optimizer_diff.state_dict()},
                           osp.join(args.snapshot_dir, 'flowdiff.pth'))

            if actual_step >= args.final_step:
                break
            cnt += 1

        scheduler.step()
        epoch_cnt += 1
        print("epoch %d, lr= %.7f" % (epoch_cnt, model.optimizer_diff.param_groups[0]["lr"]))

    print('save the final model ...')
    torch.save({'example': actual_step * args.batch_size,
                'diffusion': model.diffusion.state_dict(),
                'optimizer_diff': model.optimizer_diff.state_dict()},
               osp.join(args.snapshot_dir,
                        'flowdiff_' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))
    end = timeit.default_timer()
    print(end - start, 'seconds')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()


