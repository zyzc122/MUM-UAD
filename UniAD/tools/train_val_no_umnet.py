import argparse
import logging
import math
import os
import pprint
import shutil
import time
import timm
from torch import nn
from einops import rearrange
import constants as const
import numpy
import torch.optim
import yaml
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader
from datasets.data_builder import build_dataloader, build_dataset, build_datasets
from easydict import EasyDict

from datasets.dataset import BTADDataset
from models.model_helper import ModelHelper
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from tools.cross_attn import Cross_Attn
from tools.msnet import MSNet
from tools.pre_process import PreNet, SEblock, SENet, PreActBlock
from tools.dataset import MVTecDataset_all, MVTecDataset_val, MVTec_Path_Split, MVTecDataset_ratio
from tools.efficientnet import EfficientNet
from tools.neck import Neck
from tools.reverse_swin import swin_t
from tools.reverse_res34 import ReverseNet, BasicBlock
from tools.split_attn import Split_Attn
from utils.criterion_helper import build_criterion
from utils.dist_helper import setup_distributed
# from utils.eval_helper_attn import dump, log_metrics, merge_together, performances
# from utils.eval_helper_all import dump, log_metrics, merge_together, performances
# from utils.eval_helper_origin import dump, log_metrics, merge_together, performances
from utils.eval_helper import dump, log_metrics, merge_together, performances
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
from utils.vis_helper import visualize_compound, visualize_single, apply_ad_scoremap, normalize

import torch
parser = argparse.ArgumentParser(description="UniAD Framework")
parser.add_argument("--config", default="/home/smart-solution-server-001/Documents/anomaly_PycharmProjects/Saliency_MSTAD/experiments/MVTec-AD/config.yaml")
parser.add_argument("--type", default="normal")
parser.add_argument("--clsname", default="capsule")
parser.add_argument("-e", "--evaluate", default=False,)
parser.add_argument("--local_rank", default=None, help="local rank for dist")
import kornia


criterion_bce2 = torch.nn.BCEWithLogitsLoss(reduction='none')
mse = nn.MSELoss()

def supervised_loss(gt, pre, gt_gradient, pre_gradient):

    # pre = pre.repeat(1, gt.shape[1], 1, 1)
    # loss_ce = mse(pre, gt)
    gt = torch.clamp(gt, min=1e-7, max=1 - 1e-7)
    pre = torch.clamp(pre, min=1e-7, max=1 - 1e-7)
    gt_gradient = torch.clamp(gt_gradient, min=1e-7, max=1 - 1e-7)
    pre_gradient = torch.clamp(pre_gradient, min=1e-7, max=1 - 1e-7)
    loss_ce = criterion_bce2(input=pre, target=gt)
    loss_ce2 = criterion_bce2(input=pre_gradient, target=gt_gradient)
    # sigma_exp = torch.exp(-sigma)
    # loss = (sigma_exp) * (loss_ce + 0.1) + 0.5*sigma
    # loss = (sigma_exp) * loss_ce + sigma
    loss = loss_ce + loss_ce2
    #loss = (sigma_exp) * (loss_ce) + 0.5*torch.max(sigma, -1*torch.ones_like(sigma))

    loss = torch.mean(loss)
    return loss

def multi_uncertainty_cls_loss(gt, pre, sigma, valid_mask=None):

    pre = pre.repeat(1, gt.shape[1], 1, 1)
    # loss_ce = mse(pre, gt)
    loss_ce = criterion_bce2(input=pre, target=gt)
    sigma_exp = torch.exp(-sigma)
    # loss = (sigma_exp) * (loss_ce + 0.1) + 0.5*sigma
    # loss = (sigma_exp) * loss_ce + sigma
    loss = (sigma_exp) * loss_ce + 0.5*sigma
    #loss = (sigma_exp) * (loss_ce) + 0.5*torch.max(sigma, -1*torch.ones_like(sigma))

    if valid_mask is not None:
        loss = torch.mean(loss, [1], keepdim=True)
        loss = torch.sum(loss * valid_mask) / (torch.sum(valid_mask) + 1e-6)
    else:
        loss = torch.mean(loss)
    return loss

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
    tb_logger = SummaryWriter(config.log_path + "/events_dec/" + current_time)
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

    # prenet = PreNet(in_c=5, out_c=3)
    # prenet = SEblock(channels=3)
    # prenet = SENet(PreActBlock, [2,2,2])
    # efficientnet = EfficientNet.from_pretrained(model_name='efficientnet-b4',
    #                                             outblocks=[1, 5, 9, 21], outstrides=[2, 4, 8, 16])
    # nec = Neck()
    swin = timm.create_model("swin_base_patch4_window7_224", pretrained=True).cuda()
    baselayers = list(swin.children())
    layer0 = nn.Sequential(*baselayers[:2])
    layers = baselayers[2]
    layer1 = nn.Sequential(*layers)[0]
    layer2 = nn.Sequential(*layers)[1]
    swin_model = nn.Sequential(layer0, layer1, layer2)
    for param in swin_model.parameters():
        param.requires_grad = False

    cross_attn_net = Cross_Attn(feature_size=(14,14), feature_jitter=False,scale=20.0, prob=1.0,
                                hidden_dim=256, pos_embed_type='learned',
                                save_recon=False, initializer='xavier_uniform',
                                nhead=8,
                                num_encoder_layers=10,
                                num_decoder_layers=10,
                                dim_feedforward=1024)
    # cross_attn_net = Split_Attn(feature_size=(14,14), feature_jitter=False,scale=20.0, prob=1.0,
    #                             hidden_dim=256, pos_embed_type='learned',
    #                             save_recon=False, initializer='xavier_uniform',
    #                             nhead=8,
    #                             num_encoder_layers=10,
    #                             num_decoder_layers=10,
    #                             dim_feedforward=1024)
    # reverse_net = swin_t()
    reverse_net = ReverseNet(block_type='basic', instrides= [16], inplanes= [512])
    # weight_net = MSNet(num_branch=config.dataset.num_branch, alpha=config.dataset.msnet_alpha)
    # prenet.cuda()
    # nec.cuda()
    # efficientnet.cuda()
    swin_model.cuda()
    # cross_attn_net.cuda()
    reverse_net.cuda()
    # weight_net.cuda()
    # for param in prenet.parameters():
    #     param.requires_grad = True
    # for param in efficientnet.parameters():
    #     param.requires_grad = True
    # for param in nec.parameters():
    #     param.requires_grad = True

    layers = []
    for module in config.net:
        layers.append(module["name"])
    # layers = ['backbone', 'neck', 'reconstruction']
    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))
    logger.info("layers: {}".format(layers))
    logger.info("active layers: {}".format(active_layers))

    # for name, param in getattr(model, 'reconstruction').named_parameters():
    #     param.requires_grad = False

    # for name, param in getattr(model, 'reconstruction').named_parameters():
        # if "encoder" in name:
        #     param.requires_grad = True
        # if "enc_learned_embed" in name:
        #     param.requires_grad = True
        # elif "enc_head_attn" in name:
        #     param.requires_grad = True
        # elif "enc_norm0" in name:
        #     param.requires_grad = True
        # if "output_proj" in name:
        #     param.requires_grad = True
        # elif "input_proj1" in name:
        #     param.requires_grad = True

    #     # elif "dec_learned_embed" in name:
    #     #     param.requires_grad = True
    #     # elif "dec_head_attn" in name:
    #     #     param.requires_grad = True
    #     # elif "dec_norm" in name:
    #     #     param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    # parameters needed to be updated
    # parameters = [
    #     {"params": getattr(model, layer).parameters()} for layer in active_layers
    # ]
    # for layer in active_layers:
    #     if "enc_learned_embed" in name:
    #         param.requires_grad = True
    parameters = [
        {"params": getattr(model, layer).parameters()} for layer in active_layers
    ]
    # parameters.append({'params':cross_attn_net.parameters()})
    parameters.append({'params':reverse_net.parameters()})
    # parameters.append({'params':weight_net.parameters()})
    # parameters.append({'params':efficientnet.parameters()})
    # parameters.append({'params':nec.parameters()})
    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    # optimizer = get_optimizer(parameters, config.trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    key_metric = config.evaluator["key_metric"]
    best_metric = 0
    last_epoch = 0

    # load model: auto_resume > resume_model > load_path
    auto_resume = config.saver.get("auto_resume", True)
    resume_model = config.saver.get("resume_model", None)
    # load_path = checkpoints/ckpt.pth.tar
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

    # train_loader, val_loader = build_datasets(config.dataset, distributed=True)
    # 构建数据集

    train_all = []
    val_all = []
    best_metric = 0
    # 分离 训练和测试数据 路径
    for cls_name in const.MVTEC_CATEGORIES:
        train_image_files, val_image_files = MVTec_Path_Split(train_ratio=0.75,
                         root_image='/home/smart-solution-server-001/Documents/anomaly_PycharmProjects/MvTec',
                         category=cls_name, random_seed=235)
        train_all.extend(train_image_files)
        val_all.extend(val_image_files)

    train_set = MVTecDataset_ratio(train_all,
                                   methods= ['mstad_refine'],
                                   # ['fastflow_refine', 'edgrec_refine', "cflow",
                                   #                      'draem_refine', 'mstad_refine'],
                                   root_flabels= '/home/smart-solution-server-001/Documents/anomaly_PycharmProjects/UMNet/UMNet_train/Data/'
                                                 'MSRAB_fakelabel_refine/',
                                   input_size= 224, is_train=True)
    train_loader = DataLoader(train_set, batch_size=config.dataset.batch_size,
                              shuffle=True, num_workers=6)
    val_set = MVTecDataset_ratio(val_all,
                                 methods= ['mstad_refine'],
                                 # ['fastflow_refine', 'edgrec_refine', "cflow",
                                 #                    'draem_refine', 'mstad_refine'],
                                   root_flabels= '/home/smart-solution-server-001/Documents/anomaly_PycharmProjects/UMNet/UMNet_train/Data/'
                                                 'MSRAB_fakelabel_refine/',
                                   input_size= 224, is_train=False)
    val_loader = DataLoader(val_set, batch_size=config.dataset.batch_size,
                              shuffle=True, num_workers=6)
    # train_loader, val_loader = build_datasets(config.dataset, distributed=True)


    # if True:
    #     validate(val_loader, swin_model, model, reverse_net)
    #     return

    criterion = build_criterion(config.criterion)

    for epoch in range(last_epoch, config.trainer.max_epoch):
        # train_loader.sampler.set_epoch(epoch)
        # val_loader.sampler.set_epoch(epoch)
        last_iter = epoch * len(train_loader)
        train_one_epoch(
            swin_model,
            # weight_net,
            reverse_net,
            cross_attn_net,
            train_loader,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            last_iter,
            tb_logger,
            criterion,
            frozen_layers,
            # fig,
        )
        lr_scheduler.step(epoch)

        if (epoch + 1) % config.trainer.val_freq_epoch == 0:
            ret_metrics = validate(val_loader, swin_model, model, reverse_net)

            # ret_key_metric = ret_metrics[key_metric]
            # is_best = ret_key_metric >= best_metric
            # best_metric = max(ret_key_metric, best_metric)
            # save_checkpoint(
            #     {
            #         "epoch": epoch + 1,
            #         "arch": config.net,
            #         "state_dict": model.state_dict(),
            #         "best_metric": best_metric,
            #         "optimizer": optimizer.state_dict(),
            #     },
            #     is_best,
            #     config,
            # )


def train_one_epoch(
        swin_model,
        # weight_net,
        reverse_net,
    cross_attn_net,
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    start_iter,
    tb_logger,
    criterion,
    frozen_layers,
    # fig,
):

    batch_time = AverageMeter(config.trainer.print_freq_step)
    data_time = AverageMeter(config.trainer.print_freq_step)
    losses = AverageMeter(config.trainer.print_freq_step)

    model.train()
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

        out = input['image']
        extrac_list = []
        for i in range(len(swin_model)):
            kernel = torch.ones(2, 2).cuda()
            gradient = kornia.morphology.gradient(input['gradient'].cuda(), kernel)
            input.update({'gradient': gradient})
            out = swin_model[i](out.cuda())
            # out_ = rearrange(
            #     out, "b (h w) c -> b c h w", h=int(math.sqrt(c))
            # )  # B x C X H x W
            extrac_list.append(out)
        _, c, _ = out.size()
        out_ = rearrange(
                out, "b (h w) c -> b c h w", h=int(math.sqrt(c))
            )  # B x C X H x W
        input.update({'feature_align': out_})
        outputs = model.reconstruction(input)
        outputs.update(input)
        # outputs = model(input)
        pred = torch.sqrt(
            # torch.sum((outputs['feature_align'] - outputs['feature_rec']) ** 2, dim=1, keepdim=True)
                (outputs['feature_align'] - outputs['feature_rec']) ** 2
            )  # B x 1 x H x W
        # pred_mask, pred_list = reverse_net(torch.cat((outputs['feature_align'].cuda(), pred.cuda()), dim=1))
        pred_mask, gradient, pred_list = reverse_net(pred, extrac_list)
        # pred_mask, pred_list = reverse_net(torch.cat((outputs['feature_align'], pred), dim=1), extrac_list)
        # feature_map = prenet(input['flabels'].cuda())
        # weight_maps = weight_net(input['flabels'].cuda(), extrac_list)
        # weight_maps = weight_net(input['flabels'].cuda(), pred_list)

        # feature_map = efficientnet(input['flabels'].cuda())
        # feature_map = nec(feature_map)

        # weights = cross_attn_net(weight_maps, outputs['encoder_list'], outputs['tgt_list'])

        # flabels = process_gt(input['flabels'], valid_mask=None)
        loss = supervised_loss(input['flabels'].cuda(), pred_mask.cuda(),
                               input['gradient'].cuda(), gradient.cuda(),)
        # loss = multi_uncertainty_cls_loss(input['flabels'].cuda(), pred_mask.cuda(),
        #                                   weight_maps,
        #                                   valid_mask=None)

        # loss = 0
        # for name, criterion_loss in criterion.items():
        #     weight = criterion_loss.weight
        #     loss += weight * criterion_loss(outputs)
        # 高斯混合分布 loss
        # loss += 0.1 * outputs['KLD']
        # loss += 0.1 * outputs['loss_subspace']

        # backward
        optimizer.zero_grad()

        loss.backward()

        # update
        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)

        if (curr_step + 1) % config.trainer.print_freq_step == 0:
            tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
            tb_logger.add_scalar("lr", current_lr, curr_step + 1)
            tb_logger.flush()

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

def validate(val_loader, swin_model, model, reverse_net):
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
    with torch.no_grad():
        for i, input in enumerate(val_loader):


            # out = swin_model(input['image'].cuda())
            # b, h, c = out.size()
            # out_ = rearrange(
            #     out, "b (h w) c -> b c h w", h=int(math.sqrt(h))
            # )  # B x C X H x W
            out = input['image']
            extrac_list = []
            for i in range(len(swin_model)):
                out = swin_model[i](out.cuda())
                # _, c, _ = out.size()
                # out_ = rearrange(
                #     out, "b (h w) c -> b c h w", h=int(math.sqrt(c))
                # )  # B x C X H x W
                extrac_list.append(out)
            _, c, _ = out.size()
            out_ = rearrange(
                out, "b (h w) c -> b c h w", h=int(math.sqrt(c))
            )  # B x C X H x W
            input.update({'feature_align': out_})
            outputs = model.reconstruction(input)
            outputs.update(input)
            pred = torch.sqrt(
                # torch.sum((outputs['feature_align'] - outputs['feature_rec']) ** 2, dim=1, keepdim=True)
                (outputs['feature_align'] - outputs['feature_rec']) ** 2
            )  # B x 1 x H x W
            # pred_mask, pred_list = reverse_net(torch.cat((outputs['feature_align'].cuda(), pred.cuda()), dim=1))
            pred_mask, gradient, pred_list = reverse_net(pred, extrac_list)
            # pred_mask, pred_list = reverse_net(torch.cat((outputs['feature_align'], pred), dim=1),
            #                                    extrac_list)
            # pred_mask, pred_list = reverse_net(torch.cat((outputs['feature_align'], outputs['feature_rec']), dim=1))
            outputs.update({
                "pred": pred_mask,
            })
        #
            loss = 0
            # loss = supervised_loss(input['flabels'].cuda(), pred_mask.cuda(),
            #                    input['gradient'].cuda(), gradient.cuda(),)
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
    ret_metrics = {}  # only ret_metrics on rank0 is not empty
    logger.info("Gathering final results ...")
    # total loss
    logger.info(" * Loss {:.5f}\ttotal_num={}".format(final_loss, total_num.item()))
    # pred, masks = (1725, 224, 224); config.evaluator.eval_dir = ./result_eval_temp
    fileinfos, preds, masks = merge_together(config.evaluator.eval_dir)
    # fileinfos, preds, masks, attn_output_weights = merge_together(config.evaluator.eval_dir)
    shutil.rmtree(config.evaluator.eval_dir)
    # ret_metrics = performances(fileinfos, preds, masks, config.evaluator.metrics, attn_output_weights)
    ret_metrics = performances(fileinfos, preds, masks, config.evaluator.metrics)
    log_metrics(ret_metrics, config.evaluator.metrics)
    # print('********log_metrics:{}'.format(log_metrics))
    # 生成 视觉热力图
    if args.evaluate and config.evaluator.get("vis_compound", None):
            visualize_compound(
                fileinfos,
                preds,
                masks,
                config.evaluator.vis_compound,
                config.dataset.image_reader,
            )
    if args.evaluate and config.evaluator.get("vis_single", None):
        visualize_single(
            fileinfos,
            preds,
            config.evaluator.vis_single,
            config.dataset.image_reader,
        )
    model.train()
    return ret_metrics

if __name__ == "__main__":
    main()
