import argparse
import logging
import math
import os
import pprint
import shutil
import sys
import time
from torch import nn
from einops import rearrange
import constants as const
import numpy
import torch.optim
import yaml
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from easydict import EasyDict
from models.model_helper import ModelHelper
from tools.msnet import MSNet
from tools.dataset import MVTecDataset_all, MVTecDataset_val, MVTec_Path_Split, MVTecDataset_ratio

from tools.reverse_res34 import ReverseNet, BasicBlock
from utils.criterion_helper import build_criterion
from utils.eval_helper import dump, log_metrics, merge_together,  uni_performances
from utils.eval_helper_uniAD import dump1, merge_together1, uni_performances1
from utils.lr_helper import get_scheduler
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
    update_config,
)
from utils.optimizer_helper import get_optimizer
from thop import profile
import torch


parser = argparse.ArgumentParser(description="MUM-UAD Framework")
parser.add_argument("--config", default="/home/zyz-4090/PycharmProjects/MUM-UAD/UniAD/experiments/config.yaml")
parser.add_argument("--type", default="normal")
parser.add_argument("--clsname", default="capsule")
parser.add_argument("-e", "--evaluate", default=False,)
parser.add_argument("--local_rank", default=None, help="local rank for dist")

criterion_bce1 = torch.nn.BCELoss(reduction='none')
criterion_bce2 = torch.nn.BCEWithLogitsLoss(reduction='none')
mse = nn.MSELoss()


def multi_uncertainty_cls_loss(gt, pre, sigma):
    pre = pre.repeat(1, gt.shape[1], 1, 1)
    loss_ce = criterion_bce1(input=pre, target=gt)
    sigma_exp = torch.exp(-sigma)
    loss = (sigma_exp) * loss_ce + sigma
    loss_ = torch.mean(loss)

    return loss_

def multi_uncertainty_cls_loss_score(gt, pre,sigma):
    pre = pre.repeat(1, gt.shape[1])
    loss_ce = criterion_bce1(input=pre, target=gt)
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

def Evaluation_basemodel(model, dataloader_test):
    if not os.path.exists(config.evaluator.eval_dir):
        os.mkdir(config.evaluator.eval_dir)
    model.eval()
    with torch.no_grad():
        for i, input in enumerate(dataloader_test):
            outputs = model(input)
            dump1(config.evaluator.eval_dir, outputs)

    fileinfos, preds, masks = merge_together1(config.evaluator.eval_dir)
    shutil.rmtree(config.evaluator.eval_dir)
    det, loc, aupro = uni_performances1(fileinfos, preds, masks)
    print("统一边界的结果:")
    print(loc)
    print(det)
    print(aupro)
    return

def main():
    global args, config, key_metric, best_metric
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.port = config.get("port", None)
    # rank, world_size = setup_distributed(port=config.port)
    config = update_config(config)
    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.exp_path, config.evaluator.save_dir)
    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)

    current_time = get_current_time()
    # tb_logger = SummaryWriter(config.log_path + "/events_dec/" + current_time)
    logger = create_logger(
        "global_logger", config.log_path + "/dec_{}.log".format(current_time)
    )
    logger.info("args: {}".format(pprint.pformat(args)))
    logger.info("config: {}".format(pprint.pformat(config)))

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    # create model
    model = ModelHelper(config.net)
    model.cuda()

    reverse_net = ReverseNet(block_type='basic', instrides=[16], inplanes=[272])
    weight_net = MSNet(num_branch=config.dataset.num_branch, alpha=config.dataset.msnet_alpha)

    reverse_net.cuda()
    weight_net.cuda()

    layers = []
    for module in config.net:
        layers.append(module["name"])
    # layers = ['backbone', 'neck', 'reconstruction']
    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))
    logger.info("layers: {}".format(layers))
    logger.info("active layers: {}".format(active_layers))

    parameters = []
    parameters.append({'params': reverse_net.parameters()})
    parameters.append({'params': weight_net.parameters()})
    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    key_metric = config.evaluator["key_metric"]
    best_metric = 0
    last_epoch = 0

    auto_resume = config.saver.get("auto_resume", True)
    resume_model = config.saver.get("resume_model", None)
    load_path = config.saver.get("load_path", None)

    if resume_model and not resume_model.startswith("/"):
        resume_model = os.path.join(config.exp_path, resume_model)
    lastest_model = os.path.join(config.save_path, "ckpt.pth.tar")
    if auto_resume and os.path.exists(lastest_model):
        resume_model = lastest_model
    if resume_model:
        best_metric, last_epoch = load_state(resume_model, model, optimizer=optimizer)
    elif load_path:
        if not load_path.startswith("/"):
            load_path = os.path.join(config.exp_path, load_path)
        load_state(load_path, model)

    root_image = 'home/zyz-4090/PycharmProjects/MUM-UAD/dataset/BTAD'
    flabel_lsit = ['draem', 'edgrec', 'CAE', 'mstad']
    # flabel_lsit = ['seg_mask_134', 'seg_mask_135', 'seg_mask_136', 'seg_mask_137'],
    root_flabels = '/home/zyz-4090/PycharmProjects/MUM-UAD/flabels/flabels_btad/'
    base_model_weight_path = '/home/zyz-4090/PycharmProjects/MUM-UAD/MSTAD/experiments/checkpoint_btad/ckpt_best.pth.tar'

    train_all = []
    val_all = []
    best_metric = 0
    for cls_name in const.BTAD_CATEGORIES:
        _, val_image_files = MVTec_Path_Split(train_ratio=0.60,
                         root_image=root_image,
                         category=cls_name, random_seed=235)
        val_all.extend(val_image_files)

    for cls_name in const.BTAD_CATEGORIES:
        train_image_files, _ = MVTec_Path_Split(train_ratio=0.60,
                         root_image=root_image,
                         category=cls_name, random_seed=235)
        train_all.extend(train_image_files)

    train_set = MVTecDataset_ratio(train_all,
                                   methods=flabel_lsit,
                                   root_flabels=root_flabels,
                                   input_size=224, is_train=True)
    train_loader = DataLoader(train_set, batch_size=config.dataset.batch_size,
                              shuffle=True, num_workers=1)
    val_set = MVTecDataset_ratio(val_all,
                                 methods=flabel_lsit,
                                 root_flabels=root_flabels,
                                 input_size=224, is_train=False)
    val_loader = DataLoader(val_set, batch_size=config.dataset.batch_size,
                            shuffle=True, num_workers=1)


    criterion = build_criterion(config.criterion)
    Evaluation_basemodel(model, val_loader)

    for epoch in range(last_epoch, config.trainer.max_epoch):
        last_iter = epoch * len(train_loader)
        train_one_epoch(
            weight_net,
            reverse_net,
            train_loader,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            last_iter,
            criterion,
            frozen_layers,
            # fig,
        )
        lr_scheduler.step(epoch)

        load_state(base_model_weight_path, model)

        if (epoch + 1) % config.trainer.val_freq_epoch == 0:
            loc, det, aupro = validate(val_loader, model, reverse_net,)
            mean_ = (loc + det + aupro) / 3

            if mean_ >= best_metric:
                if len(os.listdir(const.OUTPUT_PATH)) != 0:
                    shutil.rmtree(const.OUTPUT_PATH)
                    os.mkdir(const.OUTPUT_PATH)
                best_metric = mean_
                torch.save(reverse_net.state_dict(), os.path.join(const.OUTPUT_PATH, f'reverseNet_{epoch}.pth'))
                # torch.save(resnet18_img.state_dict(), os.path.join(const.OUTPUT_PATH, 'resImg.pth'))

def calc_complex(model, input1, input2):
    # # flops, params = get_model_complexity_info(model, (C, H, W), as_strings=True, print_per_layer_stat=True)
    # # print('Flops:  ' + flops)
    # # print('Params: ' + params)
    # # stat(construct_model, (3, 256, 256))
    # input = torch.randn(1, C, H, W).to('cuda')
    flops, params = profile(model, inputs=(input1, input2))
    print(flops / 1024 / 1024 / 1024)
    print('Params: ', params / 1024 / 1024)
    return

def train_one_epoch(
        weight_net,
        reverse_net,
        # resnet18_img,
        # resnet18_flabels,
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    start_iter,
    criterion,
    frozen_layers,
    # fig,
):

    batch_time = AverageMeter(config.trainer.print_freq_step)
    data_time = AverageMeter(config.trainer.print_freq_step)
    losses = AverageMeter(config.trainer.print_freq_step)

    # model.train()
    reverse_net.train()
    weight_net.train()
    # freeze selected layers
    for layer in frozen_layers:
        module = getattr(model, layer)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    logger = logging.getLogger("global_logger")
    end = time.time()

    for i, input in enumerate(train_loader):

        curr_step = start_iter + i
        current_lr = lr_scheduler.get_last_lr()[0]
        data_time.update(time.time() - end)
        outputs = model(input)

        pred = torch.sqrt(
                (outputs['feature_align'] - outputs['feature_rec']) ** 2
            )  # B x C x H x W
        # print(pred.size())
        pred_mask, pred_list = reverse_net(pred)
        # print(pred_mask.size())

        flabels = input['flabels'].cuda()
        weight_maps = weight_net(flabels, pred_list)
        flabels = process_gt(input['flabels'], valid_mask=None) #二值化
        loss = multi_uncertainty_cls_loss(flabels.cuda(), pred_mask.cuda(), weight_maps)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # update
        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(reverse_net.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(weight_net.parameters(), max_norm)
        optimizer.step()
        batch_time.update(time.time() - end)

        logger.info(
            "Epoch: [{0}/{1}]\t"
            "Iter: [{2}/{3}]\t"
            "Loss {loss:.5f} \t"
            "LR {lr:.5f}\t".format(
                epoch + 1,
                config.trainer.max_epoch,
                curr_step + 1,
                len(train_loader) * config.trainer.max_epoch,
                loss=loss,
                lr=current_lr,
            )
        )
        end = time.time()

def validate(val_loader,model, reverse_net,):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    logger = logging.getLogger("global_logger")
    criterion = build_criterion(config.criterion)
    end = time.time()
    val_list = []
    os.makedirs(config.evaluator.eval_dir, exist_ok=True)

    # for val_path in val_path_list:
    #     clsname = val_path[0].split("/")[5]
    #     test_set = MVTecDataset_val(val_path, 224, clsname)
    #     val_list.append(test_set)
    # alltest_list = ConcatDataset(val_list)
    # val_loader = DataLoader(alltest_list, batch_size=config.dataset.batch_size, shuffle=True, num_workers=4)

    model.eval()
    reverse_net.eval()
    with torch.no_grad():
        for i, input in enumerate(val_loader):
            # out = swin_model(input['image'].cuda())
            # b, h, c = out.size()
            # out_ = rearrange(
            #     out, "b (h w) c -> b c h w", h=int(math.sqrt(h))
            # )  # B x C X H x W

            outputs = model(input)
            pred = torch.sqrt(
                # torch.sum((outputs['feature_align'] - outputs['feature_rec']) ** 2, dim=1, keepdim=True)
                (outputs['feature_align'] - outputs['feature_rec']) ** 2
            )  # B x 1 x H x W
            # pred = nn.Upsample(224)(pred)
            # pred_mask, pred_list = reverse_net(torch.cat((outputs['feature_align'].cuda(), pred.cuda()), dim=1))

            pred_mask, pred_list = reverse_net(pred)
            # torch.cuda.reset_peak_memory_stats()
            # peak_mem = torch.cuda.max_memory_allocated() / 1e9
            # print(peak_mem)
            N, _, _, _ = pred_mask.size()
            pred_mask_ = (F.avg_pool2d(pred_mask, (80, 80), stride=1).cpu().numpy())
            score_img = pred_mask_.reshape(N, -1).max(axis=1)


            # 通过pred_mask 获得异常得分
            # score_img = resnet18_img(pred_mask)
            # pred_mask, gradient, pred_list = reverse_net(
            #     torch.cat((outputs['feature_align'].cuda(), pred.cuda()), dim=1), extrac_list)
            # pred_mask, pred_list = reverse_net(torch.cat((outputs['feature_align'], pred), dim=1),
            #                                    extrac_list)
            # pred_mask, pred_list = reverse_net(torch.cat((outputs['feature_align'], outputs['feature_rec']), dim=1))
            outputs.update({
                "pred": pred_mask,
                "pred_score": score_img,
                # "pred": pred,
            })
        #
            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)
            num = len(outputs["filename"])
            losses.update(loss.item(), num)
            # 将test中的输出数据临时存入文件中
            dump(config.evaluator.eval_dir, outputs)
            # record loss
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.trainer.print_freq_step == 0:
                logger.info(
                    "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        i + 1, len(val_loader), batch_time=batch_time
                    )
                )

    # gather final results
    total_num = torch.Tensor([losses.count]).cuda()
    loss_sum = torch.Tensor([losses.avg * losses.count]).cuda()
    final_loss = loss_sum.item() / total_num.item()
    # ret_metrics = {}  # only ret_metrics on rank0 is not empty
    logger.info("Gathering final results ...")
    # total loss
    logger.info(" * Loss {:.5f}\ttotal_num={}".format(final_loss, total_num.item()))
    # pred, masks = (1725, 224, 224); config.evaluator.eval_dir = ./result_eval_temp
    fileinfos, preds, masks, img_has_anomalys, scores = merge_together(config.evaluator.eval_dir)
    # fileinfos, preds, masks = merge_together(config.evaluator.eval_dir)
    # fileinfos, preds, masks, attn_output_weights = merge_together(config.evaluator.eval_dir)
    shutil.rmtree(config.evaluator.eval_dir)
    # ret_metrics = performances(fileinfos, preds, masks, config.evaluator.metrics, attn_output_weights)
    # ret_metrics = performances(fileinfos, preds, masks, config.evaluator.metrics)
    # log_metrics(ret_metrics, config.evaluator.metrics)
    loc, det, aupro = uni_performances(fileinfos, preds, masks, config.evaluator.metrics, img_has_anomalys, scores)
    # loc, det, aupro = uni_performances(fileinfos, preds, masks)
    # logger.info("Loc: {}".format( loc))
    logger.info("Det: {} Loc: {} aupro: {}".format(det, loc, aupro))
    # print('********log_metrics:{}'.format(log_metrics))
    # 生成 视觉热力图
    # if args.evaluate and config.evaluator.get("vis_compound", None):
    #     visualize_compound(
    #         fileinfos,
    #         preds,
    #         masks,
    #         config.evaluator.vis_compound,
    #         config.dataset.image_reader,
    #     )
    # if args.evaluate and config.evaluator.get("vis_single", None):
    #     visualize_single(
    #         fileinfos,
    #         preds,
    #         config.evaluator.vis_single,
    #         config.dataset.image_reader,
    #     )
    # if True:
    #     visualize_pic(
    #         fileinfos,
    #         preds,
    #         config.evaluator.vis_single,
    #         config.dataset.image_reader_seg,
    #     )
    # model.train()
    return loc, det, aupro

if __name__ == "__main__":
    main()
