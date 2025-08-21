import shutil

import torch
import tqdm as tqdm
import numpy as np
from Saliency_MSTAD.tools import constants
from Saliency_MSTAD.tools.dataset import MVTec_Path_Split
from data_loader import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_unetskip import ReconstructiveSubNetwork
from loss import SSIM
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import kornia
from torch.utils.data import ConcatDataset
from UniAD_Gradient.utils.misc_helper import create_logger, get_current_time
import logging
from saliency_net.reverse_res34 import ReverseNet
from saliency_net.msnet import MSNet

criterion_bce2 = torch.nn.BCEWithLogitsLoss(reduction='none')
def multi_uncertainty_cls_loss(gt, pre, sigma):

    pre = pre.repeat(1, gt.shape[1], 1, 1)
    loss_ce = criterion_bce2(input=pre, target=gt)
    sigma_exp = torch.exp(-sigma)
    loss = (sigma_exp) * loss_ce + sigma
    loss_ = torch.mean(loss)
    return loss_

def process_gt(gt, lamda=None, th=0.5, valid_mask=None):
    #valid_mask = torch.ge(gt, 0).float()
    if lamda is None:
        th = th
    else:
        mean_g = torch.sum(gt * valid_mask, dim=[2, 3]) / (torch.sum(valid_mask, dim=[2, 3]) + 1e-6)
        th = mean_g * lamda
        th = th[:,:, None, None]
    gt = torch.ge(gt, th).to(torch.float32)
    return gt

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


    run_name = 'EdgRec_'
    # run_name = 'EdgRec_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+'all'+'_'

    # visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

    model = ReconstructiveSubNetwork(in_channels=1, out_channels=3)
    reverse_net = ReverseNet(block_type='basic', instrides=[32], inplanes=[3])
    weight_net = MSNet(num_branch=4, alpha=10)

    model.cuda()
    reverse_net.cuda()
    weight_net.cuda()
    # model.apply(weights_init)

    model.load_state_dict(torch.load("/home/smart-solution-server-001/Documents/anomaly_PycharmProjects/"
                                     "EdgRec-uni/EdgRec_uni.pckl", map_location='cuda:0'))
    for name, param in model.named_parameters():
        param.requires_grad = False

    # parameters = []
    # parameters.append({'params':reverse_net.parameters()})
    # parameters.append({'params':weight_net.parameters()})
    optimizer = torch.optim.Adam([
        {"params": reverse_net.parameters(), "lr": args.lr},
        {"params": weight_net.parameters(), "lr": args.lr}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

    # loss_l2 = torch.nn.modules.loss.MSELoss()
    # loss_ssim = SSIM()
    train_all = []
    test_all = []
    for cls_name in constants.MVTEC_CATEGORIES:
        train_image_files, test_image_files = MVTec_Path_Split(train_ratio=0.60,
                         root_image='/home/smart-solution-server-001/Documents/anomaly_PycharmProjects/MvTec',
                         category=cls_name, random_seed=235)
        train_all.extend(train_image_files)
        test_all.extend(test_image_files)


    dataset = MVTecDRAEMTrainDataset(train_all,
                                     args.anomaly_source_path,
                                     methods=
                                     ['fastflow', 'draem', 'edgrec', 'mstad'],
                                     root_flabels='/home/smart-solution-server-001/Documents/'
                                                  'anomaly_PycharmProjects/Saliency_MSTAD/data/flabels/',
                                     resize_shape=[256, 256])
    dataloader = DataLoader(dataset, batch_size=args.bs,
                            shuffle=True, num_workers=16)

    dataset_test = MVTecDRAEMTestDataset(test_all,
                                     ['fastflow', 'draem', 'edgrec', 'mstad'],
                                     '/home/smart-solution-server-001/Documents/'
                                                  'anomaly_PycharmProjects/Saliency_MSTAD/data/flabels/',
                                     [256, 256])
    dataloader_test = DataLoader(dataset_test, batch_size=1,
                            shuffle=True, num_workers=16)

    current_time = get_current_time()
    logger = create_logger(
        "global_logger", 'logs' + "/dec_{}.log".format(current_time)
    )
    logger = logging.getLogger("global_logger")

    reverse_net.load_state_dict(torch.load('/home/smart-solution-server-001/Documents/'
                                           'anomaly_PycharmProjects/EdgRec-uni/check/reversenet_268.pckl'))
    if True:
        auroc, auroc_pixel = val(model, reverse_net, dataset_test, dataloader_test)
        return

    n_iter = 0
    kernel = torch.ones(3, 3).cuda()
    best = 0
    for epoch in range(args.epochs):
        print("Epoch: "+str(epoch))
        start_iter = epoch * len(dataloader)
        for i_batch, sample_batched in enumerate(dataloader):
            # gray_batch = sample_batched["image"].cuda()
            # anomaly_mask = sample_batched["anomaly_mask"].cuda()
            gray_grayimage=sample_batched["auggray"].cuda()
            gradient=kornia.morphology.gradient(gray_grayimage,kernel)
            gray_rec = model(gradient)
            # 將重建圖和原圖 輸入到reversenet中
            # joined_in = torch.cat((gray_rec, sample_batched["image"].cuda()), dim=1)
            pred = torch.sqrt(
                (gray_rec - sample_batched["image"].cuda()) ** 2
            )  # B x 1 x H x W
            pred_mask, pred_list = reverse_net(pred)
            weight_maps = weight_net(sample_batched['flabels'].cuda(), pred_list)
            flabels = process_gt(sample_batched['flabels'], valid_mask=None)
            loss = multi_uncertainty_cls_loss(flabels.cuda(), pred_mask.cuda(), weight_maps)
            # weight_maps, input['gradient'].cuda(), gradient.cuda())

            # l2_loss = loss_l2(gray_rec,gray_batch)
            # ssim_loss = loss_ssim(gray_rec, gray_batch)
            # loss = l2_loss + ssim_loss
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
        OUTPUT_PATH = "/home/smart-solution-server-001/Documents/anomaly_PycharmProjects/EdgRec-uni/check/"
        if (epoch + 1) % 1 == 0:
            auroc, auroc_pixel = val(model, reverse_net, dataset_test, dataloader_test)
            mean_ = (auroc + auroc_pixel) / 2
            logger.info("Det:{} Loc:{} / Mean:{}".format(auroc, auroc_pixel, mean_))
            if mean_ > best:
                best = mean_
                if len(os.listdir()) != 0:
                    if len(os.listdir(OUTPUT_PATH)) != 0:
                        shutil.rmtree(OUTPUT_PATH)
                        os.mkdir(OUTPUT_PATH)
                    # check1 = torch.load(const.OUTPUT_PATH+"reverseNet.pth")
                    # reverse_net.load_state_dict(check1)
                    torch.save(reverse_net.state_dict(), os.path.join(OUTPUT_PATH + "/reversenet_{}.pckl".format(epoch)))
                else:
                    torch.save(reverse_net.state_dict(),
                               os.path.join(OUTPUT_PATH + "/reversenet_{}.pckl".format(epoch)))

                # torch.save(reverse_net.state_dict(), os.path.join(run_name + "reversenet.pckl"))

        # torch.save(reverse_net.state_dict(), os.path.join(run_name+"reversenet.pckl"))
        # torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))


def val(model, reverse_net, dataset_test, dataloader_test):
    obj_auroc_pixel_list = []
    obj_auroc_image_list = []


    model.eval()
    reverse_net.eval()
    # dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
    # dataloader = DataLoader(dataset, batch_size=1,
    #                         shuffle=False, num_workers=12)
    img_dim = 256
    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset_test)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset_test)))
    mask_cnt = 0

    #calculate pro
    # pro_gt=[]
    # pro_out=[]
    anomaly_score_gt = []
    anomaly_score_prediction = []

    # msgms = MSGMSLoss().cuda()
    kernel=torch.ones(3,3).cuda()

    print("==============================")
    with torch.no_grad():
        # i=0
        # if not os.path.exists(f'{savepath}/{obj_name}'):
        #     os.makedirs(f'{savepath}/{obj_name}')
        for i_batch, sample_batched in enumerate(dataloader_test):
            # gray_batch = sample_batched["image"].cuda()
            gray_gray = sample_batched["imagegray"].cuda()
            gradient = kornia.morphology.gradient(gray_gray, kernel)

            is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            gray_rec = model(gradient)
            pred = torch.sqrt(
                (gray_rec - sample_batched["image"].cuda()) ** 2
            )  # B x 1 x H x W
            pred_mask, pred_list = reverse_net(pred)

            # recimg=gray_rec.detach().cpu().numpy()[0]
            # recimg=np.transpose(recimg,(1,2,0))*180
            # recimg=recimg.astype('uint8')
            # oriimg=gray_batch.detach().cpu().numpy()[0]
            # oriimg=np.transpose(oriimg,(1,2,0))*180
            # oriimg = oriimg.astype('uint8')
            # colorD=ColorDifference(recimg,oriimg)

            #msgms
            # mgsgmsmap=msgms(gray_rec, gray_batch, as_loss=False)
            # mgsgmsmapmean = mean_smoothing(mgsgmsmap, 21)
            # out_mask_gradient = mgsgmsmapmean.detach().cpu().numpy()

            #combined
            # out_mask_averaged=colorD[None,None,:,:]+out_mask_gradient
            N, _, _, _ = pred_mask.size()
            pred_mask_ = (F.avg_pool2d(pred_mask, (80, 80), stride=1).cpu().numpy())
            image_score = pred_mask_.reshape(N, -1).max(axis=1)
            out_mask_averaged=pred_mask.cpu().numpy()
                #'''save result images
                # if saveimages:
                #     segresult=out_mask_averaged[0,0,:,:]
                #     truemaskresult=true_mask[0,0,:,:]
                #     see_img(gray_rec,f'{savepath}/{obj_name}/',i,'rec')
                #     see_img(gray_batch,f'{savepath}/{obj_name}/',i,'orig')
                #     see_img_heatmap(gray_batch,segresult,f'{savepath}/{obj_name}/',i,'hetamap')
                    # savefig(gray_batch,segresult,truemaskresult,f'{savepath}/{obj_name}/'+f'segresult{i}.png',gray_rec)
                    # i=i+1
                #'''

            # image_score = np.max(out_mask_averaged)
            anomaly_score_prediction.append(image_score)
            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_averaged.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1

        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        # ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        # ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
        # obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        # obj_ap_image_list.append(ap)
        # obj_pro_list.append(au_pro)
        # print(obj_name)
        print("AUC Image:  " +str(auroc))
        # print("AP Image:  " +str(ap))
        print("AUC Pixel:  " +str(auroc_pixel))
        # print("AP Pixel:  " +str(ap_pixel))
        # print("PRO:  " +str(au_pro))

    return auroc, auroc_pixel


    # write_results_to_file(run_name, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)


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


