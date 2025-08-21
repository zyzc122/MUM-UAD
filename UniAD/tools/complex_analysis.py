import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from thop import profile
from models.model_helper import ModelHelper
import yaml
import argparse
from easydict import EasyDict
from tools.msnet import MSNet
from tools.reverse_res34 import ReverseNet
from utils.optimizer_helper import get_optimizer
from utils.lr_helper import get_scheduler
from utils.misc_helper import update_config
import time
#
# torch.cuda.reset_peak_memory_stats()
# peak_mem = torch.cuda.max_memory_allocated() / 1e9
# print(peak_mem)
criterion_bce1 = torch.nn.BCELoss(reduction='none')
def calc_complex(model, input):
    flops, params = profile(model, inputs=input)
    print('Flops:', flops / 1024 / 1024 / 1024)
    print('Params: ', params / 1024 / 1024)
    return

def get_gpu_memory():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used // 1024**3

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

def multi_uncertainty_cls_loss(gt, pre, sigma):

    pre = pre.repeat(1, gt.shape[1], 1, 1)
    # loss_ce = mse(pre, gt)
    loss_ce = criterion_bce1(input=pre, target=gt)
    sigma_exp = torch.exp(-sigma)
    # loss = (sigma_exp) * (loss_ce + 0.1) + 0.5*sigma
    loss = (sigma_exp) * loss_ce + sigma
    # loss = (sigma_exp) * loss_ce + 0.5*sigma
    # loss_ce2 = criterion_bce2(input=pre_gradient, target=gt_gradient)
    # loss_ce2 = criterion_bce2(input=pre_gradient, target=gt_gradient)
    #loss = (sigma_exp) * (loss_ce) + 0.5*torch.max(sigma, -1*torch.ones_like(sigma))

    # if valid_mask is not None:
    #     loss = torch.mean(loss, [1], keepdim=True)
    #     loss = torch.sum(loss * valid_mask) / (torch.sum(valid_mask) + 1e-6)
    # else:
    loss_ = torch.mean(loss)

    return loss_

if __name__ == "__main__":
    # create model
    parser = argparse.ArgumentParser(description="UniAD Framework")
    parser.add_argument("--config", default="/home/zyz-4090/PycharmProjects/Saliency_MSTAD_from3090/"
                                            "Saliency_UniAD/experiments/MVTec-AD/config.yaml")
    parser.add_argument("--type", default="normal")
    parser.add_argument("--clsname", default="capsule")
    parser.add_argument("-e", "--evaluate", default=False, )
    parser.add_argument("--local_rank", default=None, help="local rank for dist")

    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.port = config.get("port", None)
    # rank, world_size = setup_distributed(port=config.port)
    config = update_config(config)

    model = ModelHelper(config.net)
    model.cuda()

    reverse_net = ReverseNet(block_type='basic', instrides=[16], inplanes=[272])
    reverse_net.cuda()
    # weight_net = MSNet(num_branch=4, alpha=10)
    # weight_net.cuda()
    # ============================================================================
    # 计算参数和Flops
    # ============================================================================
    # image = torch.randn(1, 3, 224, 224).cuda()
    # input = {"image": image, "clsname": "UniAD", "filename": None}
    # calc_complex(model, (input, ))
    # outputs = model(input)
    # pred = torch.sqrt(
    #     (outputs['feature_align'] - outputs['feature_rec']) ** 2
    # )
    # calc_complex(reverse_net, (pred,))
    # pred_mask, pred_list = reverse_net(pred)
    # flabels = torch.randn(1, 4, 224, 224).cuda()
    # calc_complex(weight_net, (flabels, pred_list))

    # ============================================================================
    # 计算Memery footprint
    # ============================================================================
    # optimizer_uniad = get_optimizer(model.parameters(), config.trainer.optimizer)
    # loss_MSTAD = torch.nn.MSELoss()
    # image = torch.randn(6, 3, 224, 224).cuda()
    # input = {"image": image}
    # flabels = process_gt(torch.randn(6, 4, 224, 224), valid_mask=None)
    #
    # torch.cuda.reset_peak_memory_stats()
    # optimizer_uniad.zero_grad()
    # outputs = model(input)
    # loss1 = loss_MSTAD(outputs['feature_align'], outputs['feature_rec'])
    # loss1.backward()
    # optimizer_uniad.step()
    # peak_mem = torch.cuda.max_memory_allocated() / 1e9
    # print(f"Train UniAD Memery:{peak_mem}GB")

    # parameters = []
    # parameters.append({'params': reverse_net.parameters()})
    # parameters.append({'params': weight_net.parameters()})
    # optimizer = get_optimizer(parameters, config.trainer.optimizer)
    # flabels = process_gt(torch.randn(6, 4, 224, 224), valid_mask=None)
    # pred = torch.randn(6, 272, 14, 14).cuda()
    #
    # torch.cuda.reset_peak_memory_stats()
    # optimizer.zero_grad()
    # # pred = torch.sqrt(
    # #     (outputs['feature_align'] - outputs['feature_rec']) ** 2
    # # )
    # pred_mask, pred_list = reverse_net(pred)
    # weight_maps = weight_net(flabels.cuda(), pred_list)
    # loss2 = multi_uncertainty_cls_loss(flabels.cuda(), pred_mask.cuda(), weight_maps)
    # loss2.backward()
    # optimizer.step()
    # peak_mem = torch.cuda.max_memory_allocated() / 1e9
    # print(f"Train MUM Memery:{peak_mem}GB")

    # ============================================================================
    # 计算推理内存
    # ============================================================================
    image = torch.randn(1, 3, 224, 224).cuda()
    input = {"image": image}
    model.eval()
    reverse_net.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        outputs = model(input)
        pred = torch.sqrt(
            (outputs['feature_align'] - outputs['feature_rec']) ** 2
        )
        pred_mask, pred_list = reverse_net(pred)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"Test MUM Memery:{peak_mem}GB")
    # print(f"Test UniAD Memery:{peak_mem}GB")

    # ============================================================================
    # 计算Fps
    # ============================================================================
    # image = torch.randn(1, 3, 224, 224).cuda()
    # input = {"image": image}
    # model.eval()
    # reverse_net.eval()
    # N = 1000
    # with torch.no_grad():
    #     outputs = model(input)
    #     pred = torch.sqrt(
    #             (outputs['feature_align'] - outputs['feature_rec']) ** 2
    #         )
    #     pred_mask, pred_list = reverse_net(pred)
    # start_time = time.time()
    # with torch.no_grad():
    #     for l in range(N):
    #         outputs = model(input)
    #         pred = torch.sqrt(
    #             (outputs['feature_align'] - outputs['feature_rec']) ** 2
    #         )
    #         pred_mask, pred_list = reverse_net(pred)
    # end_time = time.time()
    # avg_infer_time = (end_time - start_time) / N * 1000
    # print(f"inference time: {avg_infer_time} ms")




