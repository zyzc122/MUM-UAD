import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from thop import profile
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import yaml
import argparse
from easydict import EasyDict
from msnet import MSNet
from reverse_res34 import ReverseNet
from torch import optim
from loss import FocalLoss, SSIM
import time

CUDA_LAUNCH_BLOCKING = 1
criterion_bce1 = torch.nn.BCEWithLogitsLoss()
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

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.cuda()
    # reverse_net = ReverseNet(block_type='basic', instrides=[32], inplanes=[3])
    # reverse_net.cuda()
    # weight_net = MSNet(num_branch=4, alpha=10)
    # weight_net.cuda()
    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.cuda()

    # ============================================================================
    # 计算参数和Flops
    # ============================================================================

    # input = torch.randn(1, 3, 256, 256).cuda()
    # calc_complex(model, (input, ))
    # outputs = model(input)
    #
    # joined_in = torch.cat((outputs.detach(), input), dim=1)
    # calc_complex(model_seg, (joined_in,))
    #
    # pred = torch.sqrt((outputs - input) ** 2)
    # calc_complex(reverse_net, (pred,))
    # pred_mask, pred_list = reverse_net(pred)
    # flabels = torch.randn(1, 4, 256, 256).cuda()
    # calc_complex(weight_net, (flabels, pred_list))

    # ============================================================================
    # 计算Memery footprint
    # ============================================================================

    # input = torch.randn(6, 3, 256, 256).cuda()
    #
    # anomaly_mask = torch.randn(6, 1, 256, 256).cuda()
    # anomaly_mask[anomaly_mask > 0.5] = 1
    # anomaly_mask[anomaly_mask <= 0.5] = 0
    #
    # optimizer_draem = torch.optim.Adam([
    #     {"params": model.parameters(), "lr": 0.00001},
    #     {"params": model_seg.parameters(), "lr": 0.00001}])
    # loss_l2 = torch.nn.modules.loss.MSELoss()
    # loss_ssim = SSIM()
    # loss_focal = FocalLoss()
    #
    # torch.cuda.reset_peak_memory_stats()
    # optimizer_draem.zero_grad()
    # outputs = model(input)
    # joined_in = torch.cat((outputs.detach(), input), dim=1)
    # out_mask = model_seg(joined_in)
    #
    # out_mask_sm = torch.softmax(out_mask, dim=1)
    #
    # l2_loss = loss_l2(outputs, input)
    # ssim_loss = loss_ssim(outputs, input)
    #
    # segment_loss = loss_focal(out_mask_sm, anomaly_mask)
    # loss_draem = l2_loss + ssim_loss + segment_loss
    #
    # loss_draem.backward()
    # optimizer_draem.step()
    # peak_mem = torch.cuda.max_memory_allocated() / 1e9
    # print(f"Train DRAEM Memery:{peak_mem}GB")

    # optimizer = torch.optim.Adam([
    #     {"params": reverse_net.parameters(), "lr": 0.00001},
    #     {"params": weight_net.parameters(), "lr": 0.00001}])
    #
    # # outputs = model(input)
    # # pred = torch.sqrt((input - outputs) ** 2)
    # pred = torch.randn(6, 3, 256, 256).cuda()
    # flabels = process_gt(torch.randn(6, 4, 256, 256), valid_mask=None).cuda()
    #
    # torch.cuda.reset_peak_memory_stats()
    # optimizer.zero_grad()
    # pred_mask, pred_list = reverse_net(pred)
    # weight_maps = weight_net(flabels.cuda(), pred_list)
    #
    # loss2 = multi_uncertainty_cls_loss(flabels.cuda(), pred_mask.cuda(), weight_maps)
    # loss2.backward()
    # optimizer.step()
    # peak_mem = torch.cuda.max_memory_allocated() / 1e9
    # print(f"Train MUM Memery:{peak_mem}GB")


    # ============================================================================
    # 计算推理Memery footprint
    # ============================================================================
    input = torch.randn(1, 3, 256, 256).cuda()
    model.eval()
    model_seg.eval()
    # reverse_net.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        outputs = model(input)
        # pred = torch.sqrt((input - outputs) ** 2)
        # pred_mask, pred_list = reverse_net(pred)
        joined_in = torch.cat((outputs.detach(), input), dim=1)
        out_mask = model_seg(joined_in)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    # print(f"Test MUM Memery:{peak_mem}GB")
    print(f"Test Draem Memery:{peak_mem}GB")


    # ============================================================================
    # 计算Fps
    # ============================================================================
    # input = torch.randn(1, 3, 256, 256).cuda()
    # model.eval()
    # reverse_net.eval()
    # N = 1000
    # with torch.no_grad():
    #     outputs = model(input)
    #     pred = torch.sqrt((input - outputs) ** 2)
    #     pred_mask, pred_list = reverse_net(pred)
    #     # joined_in = torch.cat((outputs.detach(), input), dim=1)
    #     # out_mask = model_seg(joined_in)
    # start_time = time.time()
    # with torch.no_grad():
    #     for l in range(N):
    #         outputs = model(input)
    #         pred = torch.sqrt((input - outputs) ** 2)
    #         pred_mask, pred_list = reverse_net(pred)
    #         # joined_in = torch.cat((outputs.detach(), input), dim=1)
    #         # out_mask = model_seg(joined_in)
    # end_time = time.time()
    # avg_infer_time = (end_time - start_time) / N * 1000
    # print(f"inference time: {avg_infer_time} ms")




