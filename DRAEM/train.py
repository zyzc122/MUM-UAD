import torch
import tqdm as tqdm

from data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_unetskip import ReconstructiveSubNetwork
from loss import SSIM
import os
import kornia
from torch.utils.data import ConcatDataset
from UniAD_Gradient.utils.misc_helper import create_logger, get_current_time
import logging

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_on_device(obj_names, args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)


    run_name = 'EdgRec_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+'all'+'_'

    visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

    model = ReconstructiveSubNetwork(in_channels=1, out_channels=3)
    model.cuda()
    model.apply(weights_init)
    optimizer = torch.optim.Adam([
                                  {"params": model.parameters(), "lr": args.lr},
                                  ],weight_decay=0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()

    train_list = []
    for obj_name in obj_names:
        dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, resize_shape=[256, 256])
        train_list.append(dataset)
    allDataset = ConcatDataset(train_list)

    dataloader = DataLoader(allDataset, batch_size=args.bs,
                            shuffle=True, num_workers=16)
    current_time = get_current_time()
    logger = create_logger(
        "global_logger", 'logs' + "/dec_{}.log".format(current_time)
    )
    logger = logging.getLogger("global_logger")

    n_iter = 0
    kernel = torch.ones(3, 3).cuda()
    for epoch in range(args.epochs):
        print("Epoch: "+str(epoch))
        start_iter = epoch * len(dataloader)
        for i_batch, sample_batched in enumerate(dataloader):
            gray_batch = sample_batched["image"].cuda()
            anomaly_mask = sample_batched["anomaly_mask"].cuda()
            gray_grayimage=sample_batched["auggray"].cuda()
            gradient=kornia.morphology.gradient(gray_grayimage,kernel)
            gray_rec = model(gradient)
            l2_loss = loss_l2(gray_rec,gray_batch)
            ssim_loss = loss_ssim(gray_rec, gray_batch)
            loss = l2_loss + ssim_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curr_step = start_iter + i_batch
            if (curr_step + 1) % 10 == 0:
                logger.info(
                    "Epoch: [{0}/{1}]\t"
                    "Iter: [{2}/{3}]\t"
                    "Loss {loss:.5f} \t".format(
                        epoch + 1,
                        args.epochs,
                        curr_step + 1,
                        len(dataloader) * args.epochs,
                        loss=loss,
                    )
                )

            # if args.visualize and n_iter % 200 == 0:
            #     visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
            #     visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
            # if args.visualize and n_iter % 400 == 0:
            #     visualizer.visualize_image_batch(gradient, n_iter, image_name='batch_augmented')
            #     visualizer.visualize_image_batch(gray_batch, n_iter, image_name='batch_recon_target')
            #     visualizer.visualize_image_batch(gray_rec, n_iter, image_name='batch_recon_out')
            #     visualizer.visualize_image_batch(anomaly_mask, n_iter, image_name='mask_target')

            n_iter +=1

        scheduler.step()
        torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))


if __name__=="__main__":
    if __name__ == "__main__":
        import argparse

# --gpu_id
# 0
# --obj_id
# -1
# --lr
# 0.0001
# --bs
# 8
# --epochs
# 800
# --data_path
# /home/smart-solution-server-001/Documents/anomaly_PycharmProjects/MvTec
# --anomaly_source_path
# /home/smart-solution-server-001/Documents/anomaly_PycharmProjects/DRAEM-main/datasets/dtd/dtd-r1.0.1/dtd/images
# --log_path
# .
# --checkpoint_path
# .
# --visualize

        parser = argparse.ArgumentParser()
        parser.add_argument('--obj_id', action='store', type=int, required=True)
        parser.add_argument('--bs', action='store', type=int, required=True)
        parser.add_argument('--lr', action='store', type=float, required=True)
        parser.add_argument('--epochs', action='store', type=int, required=True)
        parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
        parser.add_argument('--data_path', action='store', type=str, required=True)
        parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
        parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
        parser.add_argument('--log_path', action='store', type=str, required=True)
        parser.add_argument('--visualize', action='store_true')

        args = parser.parse_args()

        obj_batch = [['capsule'],
                     ['bottle'],
                     ['carpet'],
                     ['leather'],
                     ['pill'],
                     ['transistor'],
                     ['tile'],
                     ['cable'],
                     ['zipper'],
                     ['toothbrush'],
                     ['metal_nut'],
                     ['hazelnut'],
                     ['screw'],
                     ['grid'],
                     ['wood']
                     ]

        if int(args.obj_id) == -1:
            obj_list = ['capsule',
                        'bottle',
                        'carpet',
                        'leather',
                        'pill',
                        'transistor',
                        'tile',
                        'cable',
                        'zipper',
                        'toothbrush',
                        'metal_nut',
                        'hazelnut',
                        'screw',
                        'grid',
                        'wood'
                        ]
            picked_classes = obj_list
        else:
            picked_classes = obj_batch[int(args.obj_id)]

        with torch.cuda.device(args.gpu_id):
            train_on_device(picked_classes, args)


