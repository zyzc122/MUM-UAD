
import logging
import shutil

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from msnet import MSNet
from reverse_res34 import ReverseNet
from Saliency_AD import constants
from Saliency_MSTAD_from3090.Saliency_MSTAD.tools.dataset import MVTec_Path_Split
from UniAD_Gradient.utils.misc_helper import get_current_time, create_logger
from data_loader import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset
from torch.utils.data import DataLoader, ConcatDataset
from torch import optim
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
import os

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

def test(model, reverse_net, dataset_test, dataloader_test):
    # obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    # obj_ap_image_list = []
    obj_auroc_image_list = []

    img_dim = 256
    model.eval()
    reverse_net.eval()

    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset_test)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset_test)))
    mask_cnt = 0

    anomaly_score_gt = []
    anomaly_score_prediction = []

    # display_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
    # display_gt_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
    # display_out_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
    # display_in_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
    # cnt_display = 0
    # display_indices = np.random.randint(len(dataloader_test), size=(16,))

    print("==============================")
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader_test):

            gray_batch = sample_batched["image"].cuda()
            is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            gray_rec = model(gray_batch)
            pred = torch.sqrt(
                (gray_rec - sample_batched["image"].cuda()) ** 2
            )  # B x 1 x H x W
            pred_mask, pred_list = reverse_net(pred)
            N, _, _, _ = pred_mask.size()
            pred_mask_ = (F.avg_pool2d(pred_mask, (80, 80), stride=1).cpu().numpy())
            image_score = pred_mask_.reshape(N, -1).max(axis=1)
            out_mask_averaged = pred_mask.cpu().numpy()
            # out_mask = model_seg(joined_in)
            # out_mask_sm = torch.softmax(out_mask, dim=1)


            # out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()

            # out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
            #                                                    padding=21 // 2).cpu().detach().numpy()
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
        # print(obj_name)
        print("AUC Image:  " +str(auroc))
        # print("AP Image:  " +str(ap))
        print("AUC Pixel:  " +str(auroc_pixel))
        # print("AP Pixel:  " +str(ap_pixel))
        print("==============================")
    return auroc, auroc_pixel
def train_on_device(obj_names):

    # if not os.path.exists(args.checkpoint_path):
    #     os.makedirs(args.checkpoint_path)
    # checkpoint_path = "/home/smart-solution-server-003/anomaly_PycharmProjects/DRAEM-uni/checkpoint/"
    # if not os.path.exists(args.log_path):
    #     os.makedirs(args.log_path)

    # run_name = 'DRAEM_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+'all'+'_'
    # visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    reverse_net = ReverseNet(block_type='basic', instrides=[32], inplanes=[3])
    # weight_net = MSNet(num_branch=4, alpha=10)

    model.cuda()
    reverse_net.cuda()
    # weight_net.cuda()

    model.load_state_dict(torch.load("/home/smart-solution-server-003/anomaly_PycharmProjects/DRAEM-uni/checkpoint/all.pckl"))
    reverse_net.load_state_dict(torch.load('/home/smart-solution-server-003/anomaly_PycharmProjects/'
                                           'DRAEM-uni/saliency_check/reversenet_16.pckl'))
    # model.apply(weights_init)
    for name, param in model.named_parameters():
        param.requires_grad = False

    # model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    # model_seg.cuda()
    # model_seg.load_state_dict(torch.load("/home/smart-solution-server-003/anomaly_PycharmProjects/DRAEM-uni/checkpoint/all_seg.pckl"))
    # for name, param in model_seg.named_parameters():
    #     param.requires_grad = False


    # optimizer = torch.optim.Adam([
    #                               {"params": reverse_net.parameters(), "lr": args.lr},
    #                               {"params": weight_net.parameters(), "lr": args.lr}])

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
    #                                            [800*0.8,800*0.9],gamma=0.2, last_epoch=-1)

    # loss_l2 = torch.nn.modules.loss.MSELoss()
    # loss_ssim = SSIM()
    # loss_focal = FocalLoss()

    # test_list = []
    # for obj_name in obj_names:
    #     img_dim = 256
    #     dataset = MVTecDRAEMTestDataset("/home/smart-solution-server-003/anomaly_PycharmProjects/MvTec/" +
    #                                     obj_name + "/test/", resize_shape=[img_dim, img_dim])
    #     test_list.append(dataset)
    # allDataset_test = ConcatDataset(test_list)
    # dataloader_test = DataLoader(allDataset_test, batch_size=6,
    #                         shuffle=False, num_workers=6)

    train_all = []
    test_all = []
    for cls_name in constants.MVTEC_CATEGORIES:
        train_image_files, test_image_files = MVTec_Path_Split(train_ratio=0.60,
                                                               root_image='/home/smart-solution-server-003/anomaly_PycharmProjects/MvTec',
                                                               category=cls_name, random_seed=235)
        train_all.extend(train_image_files)
        test_all.extend(test_image_files)

    # dataset = MVTecDRAEMTrainDataset(train_all,
    #                                  args.anomaly_source_path,
    #                                  methods=
    #                                  ['fastflow', 'draem', 'edgrec', 'mstad'],
    #                                  root_flabels='/home/smart-solution-server-003/anomaly_PycharmProjects/'
    #                                               'Saliency_MSTAD_from3090/Saliency_MSTAD/data/flabels/',
    #                                  resize_shape=[256, 256])
    # dataloader_train = DataLoader(dataset, batch_size=args.bs,
    #                         shuffle=True, num_workers=16)

    dataset_test = MVTecDRAEMTestDataset(test_all,
                                         [256, 256])
    dataloader_test = DataLoader(dataset_test, batch_size=1,
                                 shuffle=True, num_workers=6)

    # train_list = []
    # for obj_name in obj_names:
    #     dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, resize_shape=[256, 256])
    #     train_list.append(dataset)
    # allDataset = ConcatDataset(train_list)
    #
    # dataloader = DataLoader(allDataset, batch_size=args.bs,
    #                             shuffle=True, num_workers=6)
    current_time = get_current_time()
    logger = create_logger(
        "global_logger", '/home/smart-solution-server-003/anomaly_PycharmProjects/DRAEM-uni/logs' + "/dec_{}.log".format(current_time)
    )
    logger = logging.getLogger("global_logger")

    if True:
        det, loc = test(model, reverse_net, dataset_test, dataloader_test)
        return

    best = 0
    n_iter = 0
    for epoch in range(args.epochs):
        print("Epoch: "+str(epoch))
        start_iter = epoch * len(dataloader_train)
        for i_batch, sample_batched in enumerate(dataloader_train):
            # gray_batch = sample_batched["image"].cuda()
            aug_gray_batch = sample_batched["augmented_image"].cuda()
            # anomaly_mask = sample_batched["anomaly_mask"].cuda()

            gray_rec = model(aug_gray_batch)
            # joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

            pred = torch.sqrt(
                (gray_rec - sample_batched["image"].cuda()) ** 2
            )  # B x 1 x H x W
            pred_mask, pred_list = reverse_net(pred)
            weight_maps = weight_net(sample_batched['flabels'].cuda(), pred_list)
            flabels = process_gt(sample_batched['flabels'], valid_mask=None)
            loss = multi_uncertainty_cls_loss(flabels.cuda(), pred_mask.cuda(), weight_maps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if args.visualize and n_iter % 200 == 0:
            #     visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
            #     visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
            #     visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')
            # if args.visualize and n_iter % 400 == 0:
            #     t_mask = out_mask_sm[:, 1:, :, :]
            #     visualizer.visualize_image_batch(aug_gray_batch, n_iter, image_name='batch_augmented')
            #     visualizer.visualize_image_batch(gray_batch, n_iter, image_name='batch_recon_target')
            #     visualizer.visualize_image_batch(gray_rec, n_iter, image_name='batch_recon_out')
            #     visualizer.visualize_image_batch(anomaly_mask, n_iter, image_name='mask_target')
            #     visualizer.visualize_image_batch(t_mask, n_iter, image_name='mask_out')
            curr_step = start_iter + i_batch
            if (curr_step + 1) % 10 == 0:
                logger.info(
                    "Epoch: [{0}/{1}]\t"
                    "Iter: [{2}/{3}]\t"
                    "Loss {loss:.5f} \t".format(
                        epoch + 1,
                        args.epochs,
                        curr_step + 1,
                        len(dataloader_train) * args.epochs,
                        loss=loss,
                    )
                )
            n_iter +=1

        scheduler.step()
        OUTPUT_PATH = "/home/smart-solution-server-003/anomaly_PycharmProjects/DRAEM-uni/saliency_check/"
        if (epoch + 1) % 1 == 0:
            auroc, auroc_pixel = test(model, reverse_net, dataset_test, dataloader_test)
            mean_ = (auroc + auroc_pixel) / 2
            logger.info("Det:{} Loc:{} / Mean:{}".format(auroc, auroc_pixel, mean_))
            if mean_ > best:
                best = mean_
                if len(os.listdir()) != 0:
                    if len(os.listdir(OUTPUT_PATH)) != 0:
                        shutil.rmtree(OUTPUT_PATH)
                        os.mkdir(OUTPUT_PATH)
                    torch.save(reverse_net.state_dict(),
                               os.path.join(OUTPUT_PATH + "/reversenet_{}.pckl".format(epoch)))
                else:
                    torch.save(reverse_net.state_dict(),
                               os.path.join(OUTPUT_PATH + "/reversenet_{}.pckl".format(epoch)))


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--obj_id', action='store', type=int, required=True)
    # parser.add_argument('--bs', action='store', type=int, required=True)
    # parser.add_argument('--lr', action='store', type=float, required=True)
    # parser.add_argument('--epochs', action='store', type=int, required=True)
    # parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    # parser.add_argument('--data_path', action='store', type=str, required=True)
    # parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    # parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    # parser.add_argument('--log_path', action='store', type=str, required=True)
    # parser.add_argument('--visualize', action='store_true')

    # args = parser.parse_args()

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

    # if int(args.obj_id) == -1:
    #     obj_list = ['capsule',
    #                  'bottle',
    #                  'carpet',
    #                  'leather',
    #                  'pill',
    #                  'transistor',
    #                  'tile',
    #                  'cable',
    #                  'zipper',
    #                  'toothbrush',
    #                  'metal_nut',
    #                  'hazelnut',
    #                  'screw',
    #                  'grid',
    #                  'wood'
    #                  ]
    #     picked_classes = obj_list
    # else:
    #     picked_classes = obj_batch[int(args.obj_id)]

    # with torch.cuda.device(args.gpu_id):
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
    train_on_device(obj_list,)

