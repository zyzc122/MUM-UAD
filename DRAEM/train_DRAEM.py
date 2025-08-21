import logging

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from UniAD_Gradient.utils.misc_helper import get_current_time, create_logger, set_random_seed
from data_loader import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset
from torch.utils.data import DataLoader, ConcatDataset
from torch import optim
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
import os
from metrics import compute_pro, trapezoid

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

def test(allDataset_test, dataloader_test, model, model_seg):
    # obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    # obj_ap_image_list = []
    obj_auroc_image_list = []

    img_dim = 256
    model.eval()
    model_seg.eval()

    # dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
    # dataloader = DataLoader(dataset, batch_size=1,
    #                         shuffle=False, num_workers=0)

    total_pixel_scores = np.zeros((img_dim * img_dim * len(allDataset_test)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(allDataset_test)))
    mask_cnt = 0

    anomaly_score_gt = []
    anomaly_score_prediction = []

    display_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
    display_gt_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
    display_out_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
    display_in_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
    cnt_display = 0
    display_indices = np.random.randint(len(dataloader_test), size=(16,))

    print("==============================")
    pro_mask_list = []
    pro_pred_list = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader_test):

            gray_batch = sample_batched["image"].cuda()

            is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]

            pro_mask_list.append(true_mask.squeeze(1)[0, :, :].unsqueeze(0).numpy())

            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            gray_rec = model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)


            if i_batch in display_indices:
                t_mask = out_mask_sm[:, 1:, :, :]
                display_images[cnt_display] = gray_rec[0]
                display_gt_images[cnt_display] = gray_batch[0]
                display_out_masks[cnt_display] = t_mask[0]
                display_in_masks[cnt_display] = true_mask[0]
                cnt_display += 1


            out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()

            pro_pred_list.append(out_mask_sm[0 ,1 ,: ,:].unsqueeze(0).detach().cpu().numpy())

            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged)

            anomaly_score_prediction.append(image_score)

            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
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

        all_fprs, all_pros = compute_pro(
            anomaly_maps=np.concatenate(pro_pred_list),
            ground_truth_maps=np.concatenate(pro_mask_list)
        )
        aupro = trapezoid(all_fprs, all_pros)

        print("AUC Image:  " +str(auroc))
        # print("AP Image:  " +str(ap))
        print("AUC Pixel:  " +str(auroc_pixel))
        print("Aupro Pixel:  " +str(aupro))
        # print("AP Pixel:  " +str(ap_pixel))
        print("==============================")
    return auroc, auroc_pixel, aupro
def train_on_device(obj_names, args):

    # if not os.path.exists(args.checkpoint_path):
    #     os.makedirs(args.checkpoint_path)
    checkpoint_path = "/home/smart-solution-server-003/anomaly_PycharmProjects/DRAEM-uni/checkpoint/"
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # run_name = 'DRAEM_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+'all'+'_'
    # visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.cuda()
    model.load_state_dict(torch.load("/home/smart-solution-server-003/anomaly_PycharmProjects/DRAEM-uni/checkpoint/all.pckl"))
    # model.apply(weights_init)

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.cuda()
    model_seg.load_state_dict(torch.load("/home/smart-solution-server-003/anomaly_PycharmProjects/DRAEM-uni/checkpoint/all_seg.pckl"))


    optimizer = torch.optim.Adam([
                                  {"params": model.parameters(), "lr": args.lr},
                                  {"params": model_seg.parameters(), "lr": args.lr}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    loss_focal = FocalLoss()

    test_list = []
    for obj_name in obj_names:
        img_dim = 256
        dataset = MVTecDRAEMTestDataset("/home/smart-solution-server-003/anomaly_PycharmProjects/MvTec/" +
                                        obj_name + "/test/", resize_shape=[img_dim, img_dim])
        test_list.append(dataset)
    allDataset_test = ConcatDataset(test_list)
    dataloader_test = DataLoader(allDataset_test, batch_size=1,
                            shuffle=False, num_workers=6)

    train_list = []
    for obj_name in obj_names:
        dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, resize_shape=[256, 256])
        train_list.append(dataset)
    allDataset = ConcatDataset(train_list)

    dataloader = DataLoader(allDataset, batch_size=10,
                                shuffle=True, num_workers=6)
    current_time = get_current_time()
    logger = create_logger(
        "global_logger", '/home/smart-solution-server-003/anomaly_PycharmProjects/DRAEM-uni/logs' + "/dec_{}.log".format(current_time)
    )
    logger = logging.getLogger("global_logger")

    if True:
        det, loc, aupro = test(allDataset_test, dataloader_test, model, model_seg,)
        return

    best = 0
    n_iter = 0
    for epoch in range(args.epochs):

        model.train()
        model_seg.train()
        print("Epoch: "+str(epoch))
        start_iter = epoch * len(dataloader)
        for i_batch, sample_batched in enumerate(dataloader):
            gray_batch = sample_batched["image"].cuda()
            aug_gray_batch = sample_batched["augmented_image"].cuda()
            anomaly_mask = sample_batched["anomaly_mask"].cuda()

            gray_rec = model(aug_gray_batch)
            joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            l2_loss = loss_l2(gray_rec,gray_batch)
            ssim_loss = loss_ssim(gray_rec, gray_batch)

            segment_loss = loss_focal(out_mask_sm, anomaly_mask)
            loss = l2_loss + ssim_loss + segment_loss

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
                        len(dataloader) * args.epochs,
                        loss=loss,
                    )
                )
            n_iter +=1

        scheduler.step()

        if (epoch + 1) % 15 == 0:
            det, loc = test(allDataset_test, dataloader_test, model, model_seg,)
            mean_ = (loc+det) / 2
            logger.info("Det: {} Loc: {} / Mean: {}".format(det, loc, mean_))
            if mean_ >= best:
                best = mean_
                torch.save(model.state_dict(), os.path.join(checkpoint_path+'all'+".pckl"))
                torch.save(model_seg.state_dict(), os.path.join(checkpoint_path+'all'+"_seg.pckl"))



if __name__=="__main__":
    import argparse

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

    set_random_seed(seed=235, reproduce=False)

    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args)

