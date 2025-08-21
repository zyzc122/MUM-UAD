import glob
import logging
import shutil
import random
from metrics import compute_pro, trapezoid
import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from msnet import MSNet
from reverse_res34 import ReverseNet
from utils import constants
from utils.dataset import MVTec_Path_Split
from utils.misc_helper import get_current_time, create_logger
from data_loader import  MVTecDRAEMTrainDataset_Saliency, MVTecDRAEMTestDataset_Saliency
from torch.utils.data import DataLoader
from torch import optim
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os

def set_random_seed(seed=233, reproduce=False):
    np.random.seed(seed)
    torch.manual_seed(seed ** 2)
    torch.cuda.manual_seed(seed ** 3)
    random.seed(seed ** 4)

    if reproduce:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


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
        th = th[:, :, None, None]
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

def test(model, reverse_net, dataset_test, dataloader_test, vis=False, ev=True):
    obj_auroc_pixel_list = []
    obj_auroc_image_list = []

    img_dim = 256
    model.eval()
    reverse_net.eval()

    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset_test)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset_test)))
    mask_cnt = 0

    anomaly_score_gt = []
    anomaly_score_prediction = []

    temp_path = "/home/zyz-4090/PycharmProjects/MUM-UAD/DRAEM/temp_path/"

    print("==============================")
    pro_mask_list = []
    pro_pred_list = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader_test):
            gray_batch = sample_batched["image"].cuda()
            is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            pro_mask_list.append(true_mask.squeeze(1)[0, :, :].unsqueeze(0).numpy())
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            gray_rec = model(gray_batch)
            if ev:
                pred = torch.sqrt(
                    (gray_rec - sample_batched["image"].cuda()) ** 2
                )  # B x 1 x H x W
                pred_mask, pred_list = reverse_net(pred)
                N, _, _, _ = pred_mask.size()
                pred_mask_ = (F.avg_pool2d(pred_mask, (80, 80), stride=1).cpu().numpy())
                image_score = pred_mask_.reshape(N, -1).max(axis=1)
                out_mask_averaged = pred_mask.cpu().numpy()
            else:
                pred = torch.cat((gray_rec.detach(), gray_batch), dim=1)
                pred_mask = reverse_net(pred)
                pred_mask = torch.softmax(pred_mask, dim=1)[:, 1:, :, :]
                N, _, _, _ = pred_mask.size()
                pred_mask_ = pred_mask.cpu().numpy()
                image_score = pred_mask_.reshape(N, -1).max(axis=1)
                out_mask_averaged = pred_mask.cpu().numpy()

            pro_pred_list.append(pred_mask.squeeze(1).detach().cpu().numpy())


            sample_batched.update({
                'pred': pred_mask
            })

            if vis:
                os.makedirs(temp_path, exist_ok=True)
                dump(temp_path, sample_batched)

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

        total_gt_pixel_scores[total_gt_pixel_scores > 0] = 1
        total_gt_pixel_scores[total_gt_pixel_scores <= 0] = 0
        # total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
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

        print("AUC Image:  " + str(auroc))
        # print("AP Image:  " +str(ap))
        print("AUC Pixel:  " + str(auroc_pixel))

        print("aupro:  " +str(aupro))
        # print("AP Pixel:  " +str(ap_pixel))

        if vis:
            fileinfos, preds, masks = merge_together(temp_path)
            visualize_compound(
                fileinfos, preds, masks
            )
            shutil.rmtree(temp_path)
        print("==============================")

        reverse_net.train()
    return auroc, auroc_pixel, aupro

def train_on_device(args):
    train_all = []
    test_all = []
    for cls_name in constants.BTAD_CATEGORIES:
        train_image_files, test_image_files = MVTec_Path_Split(train_ratio=0.60,
                                                               root_image=args.root_image,
                                                               category=cls_name, random_seed=23)
        train_all.extend(train_image_files)
        test_all.extend(test_image_files)

    dataset = MVTecDRAEMTrainDataset_Saliency(train_all,
                                              None,
                                              methods=args.flabel_lsit,
                                              root_flabels=args.root_flabels,
                                              resize_shape=[256, 256])

    dataloader_train = DataLoader(dataset, batch_size=args.bs,
                                  shuffle=True, num_workers=16)

    dataset_test = MVTecDRAEMTestDataset_Saliency(test_all, [256, 256])

    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=16)

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    reverse_net = ReverseNet(block_type='basic', instrides=[32], inplanes=[3])
    weight_net = MSNet(num_branch=4, alpha=10)

    model.load_state_dict(torch.load(args.base_model_weight_path))
    model.cuda()
    model.eval()
    reverse_net.cuda()
    weight_net.cuda()

    for name, param in model.named_parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam([
                                  {"params": reverse_net.parameters(), "lr": args.lr},
                                  {"params": weight_net.parameters(), "lr": args.lr}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs*0.8, args.epochs*0.9], gamma=0.2, last_epoch=-1)

    if True:
        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.cuda()
        model_seg.load_state_dict(torch.load(args.seg_model_weight_path))
        for name, param in model_seg.named_parameters():
            param.requires_grad = False
        model_seg.eval()
        det, loc, aupro = test(model, model_seg, dataset_test, dataloader_test, False, False)
        model_seg = None

    current_time = get_current_time()
    logger = create_logger(
        "global_logger", args.log_path + "/dec_{}.log".format(current_time)
    )
    logger = logging.getLogger("global_logger")


    best = 0
    n_iter = 0
    for epoch in range(args.epochs):
        reverse_net.train()
        weight_net.train()

        print("Epoch: " + str(epoch))
        start_iter = epoch * len(dataloader_train)
        for i_batch, sample_batched in enumerate(dataloader_train):
            gray_batch = sample_batched["image"].cuda()

            for name, param in model.named_parameters():
                param.requires_grad = False

            with torch.no_grad():
                gray_rec = model(gray_batch)
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

            model.load_state_dict(torch.load(args.base_model_weight_path))


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
            n_iter += 1

        scheduler.step()

        if (epoch + 1) % 1 == 0:
            auroc, auroc_pixel, aupro = test(model, reverse_net, dataset_test, dataloader_test)
            mean_ = (auroc + auroc_pixel + aupro) / 3
            logger.info("Det:{} Loc:{} AUPRO:{}/ Mean:{}".format(auroc, auroc_pixel, aupro, mean_))
            if mean_ > best:
                best = mean_
                if len(os.listdir()) != 0:
                    if len(os.listdir(args.OUTPUT_PATH)) != 0:
                        shutil.rmtree(args.OUTPUT_PATH)
                        os.mkdir(args.OUTPUT_PATH)
                    torch.save(reverse_net.state_dict(), os.path.join(args.OUTPUT_PATH + "/reversenet_{}.pckl".format(epoch)))
                else:
                    torch.save(reverse_net.state_dict(), os.path.join(args.OUTPUT_PATH + "/reversenet_{}.pckl".format(epoch)))

def merge_together(save_dir):
    npz_file_list = glob.glob(os.path.join(save_dir, "*.npz"))
    fileinfos = []
    preds = []
    masks = []
    for npz_file in npz_file_list:
        npz = np.load(npz_file)
        fileinfos.append(
            {
                "filename": str(npz["filename"]),
                "height": npz["height"],
                "width": npz["width"],
                "clsname": str(npz["clsname"]),
            }
        )
        preds.append(npz["pred"])
        masks.append(npz["mask"])

    preds = np.concatenate(np.asarray(preds), axis=0)  # N x H x W
    masks = np.concatenate(np.asarray(masks), axis=0)  # N x H x W
    # scores = np.concatenate(np.asarray(scores), axis=0)  # N x H x W
    return fileinfos, preds, masks

def dump(save_dir, outputs):
    filenames = outputs["filename"]
    batch_size = len(filenames)
    # pred_score = outputs["pred_score"]  # B x 1 x H x W
    # pred_score = outputs["pred_score"].cpu().numpy()  # B x 1 x H x W
    # img_has_anomalys = outputs["img_has_anomaly"].cpu().numpy()  # B x 1 x H x W
    preds = outputs["pred"].cpu().numpy()  # B x 1 x H x W
    masks = outputs["mask"].cpu().numpy()  # B x 1 x H x W
    heights = outputs["height"].cpu().numpy()
    widths = outputs["width"].cpu().numpy()
    clsnames = outputs["clsname"]
    for i in range(batch_size):
        file_dir, filename = os.path.split(filenames[i])
        _, subname = os.path.split(file_dir)
        filename = "{}_{}_{}".format(clsnames[i], subname, filename)
        filename, _ = os.path.splitext(filename)
        save_file = os.path.join(save_dir, filename + ".npz")
        np.savez(
            save_file,
            filename=filenames[i],
            # score=pred_score[i],
            # img_has_anomalys=img_has_anomalys[i],
            pred=preds[i],
            mask=masks[i],
            height=heights[i],
            width=widths[i],
            clsname=clsnames[i],
        )

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def apply_ad_scoremap(image, scoremap, alpha=0.7):
    np_image = np.asarray(image, dtype=float)
    # np_image_ = np_image[:, :, ::-1]
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    # scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def visualize_compound(fileinfos, preds, masks):
    vis_dir = "/home/smart-solution-server-003/anomaly_PycharmProjects/DRAEM-uni/vis/"
    os.makedirs(vis_dir, exist_ok=True)
    # vis_dir = cfg_vis.save_dir

    # max_score = cfg_vis.get("max_score", None)
    # min_score = cfg_vis.get("min_score", None)
    # max_score = preds.max() if not max_score else max_score
    # min_score = preds.min() if not min_score else min_score

    # image_reader = build_image_reader(cfg_reader)

    for i, fileinfo in enumerate(fileinfos):
        clsname = fileinfo["clsname"]
        filename = fileinfo["filename"]
        filedir, filename = os.path.split(filename)
        _, defename = os.path.split(filedir)
        save_dir = os.path.join(vis_dir, clsname, defename)
        # save_dir = '/home/smart-solution-server-001/Documents/anomaly_PycharmProjects/Saliency_MSTAD/tools/{}'.format(
        #     save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # read image
        # h, w = int(fileinfo["height"]), int(fileinfo["width"])
        image = cv2.imread(fileinfo["filename"])
        # image = image_reader(fileinfo["filename"])
        pred = preds[i][:, :, None].repeat(3, 2)
        w, h, _ = image.shape
        pred = cv2.resize(pred, (w, h))

        # self normalize just for analysis
        scoremap_self = apply_ad_scoremap(image, normalize(pred))
        # global normalize
        pred = np.clip(pred, pred.min(), pred.max())
        pred = normalize(pred, pred.min(), pred.max())
        scoremap_global = apply_ad_scoremap(image, pred)

        if masks is not None:
            mask = (masks[i] * 255).astype(np.uint8)[:, :, None].repeat(3, 2)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            save_path = os.path.join(save_dir, filename)
            if mask.sum() == 0:
                scoremap = np.vstack([image, scoremap_global])
            else:
                scoremap = np.vstack([image, mask, scoremap_self])
        else:
            scoremap = np.vstack([image, scoremap_self])

        # scoremap = cv2.cvtColor(scoremap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, scoremap)

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=False, default=-1)
    parser.add_argument('--bs', action='store', type=int, required=False, default=6)
    parser.add_argument('--lr', action='store', type=float, required=False, default=0.0001)
    parser.add_argument('--epochs', action='store', type=int, required=False, default=100)
    parser.add_argument('--gpu_id', action='store', type=int, required=False, default=0)
    parser.add_argument('--root_image', action='store', type=str, required=False,
                        default='/home/zyz-4090/PycharmProjects/MUM-UAD/dataset/BTAD')
    parser.add_argument('--root_flabels', action='store', required=False,
                        default='/home/zyz-4090/PycharmProjects/MUM-UAD/flabels/flabels_btad/')
    parser.add_argument('--flabel_lsit', action='store', type=list, required=False,
                        default=['draem', 'edgrec', 'CAE', 'mstad'])
    parser.add_argument('--OUTPUT_PATH', action='store', type=str, required=False,
                        default="/home/zyz-4090/PycharmProjects/MUM-UAD/DRAEM/saliency_check/")
    parser.add_argument('--base_model_weight_path', action='store', type=str, required=False,
                        default='/home/zyz-4090/PycharmProjects/MUM-UAD/DRAEM/checkpoint/BTAD/DRAEM_test_0.0001_1000_bs16_266.pckl')
    parser.add_argument('--seg_model_weight_path', action='store', type=str, required=False,
                        default='/home/zyz-4090/PycharmProjects/MUM-UAD/DRAEM/checkpoint/BTAD/DRAEM_test_0.0001_1000_bs16_266_seg.pckl')
    parser.add_argument('--log_path', action='store', type=str, required=False,
                        default='/home/zyz-4090/PycharmProjects/MUM-UAD/DRAEM/logs/')
    parser.add_argument('--visualize', action='store_true')


    args = parser.parse_args()


    set_random_seed(seed=235, reproduce=False)

    with torch.cuda.device(args.gpu_id):
        train_on_device(args)

