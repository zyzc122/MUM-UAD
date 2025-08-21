import glob
import logging
import os
from easydict import EasyDict as edict
import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from sklearn import metrics
from .metrics import compute_pro, trapezoid
import numpy as np
from sklearn.metrics import auc, precision_recall_curve

def compute_aupro(preds, masks):
    """
    Compute Area Under PRO Curve (AUPRO) for anomaly segmentation.

    Args:
        preds (np.ndarray): Predicted anomaly scores (N x H x W), higher values indicate anomalies.
        masks (np.ndarray): Ground truth binary masks (N x H x W), 1=anomaly, 0=normal.

    Returns:
        float: AUPRO score (higher is better).
    """
    # Flatten predictions and masks
    flat_preds = preds.ravel()
    flat_masks = masks.ravel().astype(bool)

    # Compute Precision-Recall curve
    precision, recall, _ = precision_recall_curve(flat_masks, flat_preds)

    # Compute area under the curve (AUPRO)
    aupro = auc(recall, precision)

    return aupro
def dump1(save_dir, outputs):
    filenames = outputs["filename"]
    batch_size = len(filenames)
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
            pred=preds[i],
            mask=masks[i],
            height=heights[i],
            width=widths[i],
            clsname=clsnames[i],
        )


def merge_together1(save_dir):
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
    return fileinfos, preds, masks


class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )


class EvalDataMeta:
    def __init__(self, preds, masks):
        self.preds = preds  # N x H x W
        self.masks = masks  # N x H x W


class EvalImage:
    def __init__(self, data_meta, **kwargs):
        self.preds = self.encode_pred(data_meta.preds, **kwargs)
        self.masks = self.encode_mask(data_meta.masks)
        self.preds_good = sorted(self.preds[self.masks == 0], reverse=True)
        self.preds_defe = sorted(self.preds[self.masks == 1], reverse=True)
        self.num_good = len(self.preds_good)
        self.num_defe = len(self.preds_defe)

    @staticmethod
    def encode_pred(preds):
        raise NotImplementedError

    def encode_mask(self, masks):
        N, _, _ = masks.shape
        masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(int)  # (N, )
        return masks

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc


class EvalImageMean(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).mean(axis=1)  # (N, )


class EvalImageStd(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).std(axis=1)  # (N, )


class EvalImageMax(EvalImage):
    @staticmethod
    def encode_pred(preds, avgpool_size):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        preds = (
            F.avg_pool2d(preds, avgpool_size, stride=1).cpu().numpy()
        )  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )


class EvalPerPixelAUC:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc


eval_lookup_table = {
    "mean": EvalImageMean,
    "std": EvalImageStd,
    "max": EvalImageMax,
    "pixel": EvalPerPixelAUC,
}

def uni_performances1(fileinfos, preds, masks):
    ret_metrics = {}
    # clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    # for clsname in clsnames:
    # preds_cls = []
    # masks_cls = []
    #     for fileinfo, pred, mask in zip(fileinfos, preds, masks):
    #         # 取出test中 某一类的全部图片
    #         if fileinfo["clsname"] == clsname:
    #             preds_cls.append(pred[None, ...])
    #             masks_cls.append(mask[None, ...])
    # preds_cls = np.concatenate(np.asarray(preds), axis=0)  # N x H x W
    # masks_cls = np.concatenate(np.asarray(masks), axis=0)  # N x H x W
    data_meta = EvalDataMeta(preds, masks)
    kwargs = edict()
    kwargs.avgpool_size = [16, 16]
    eval_method_det = eval_lookup_table["max"](data_meta,  **kwargs)
    eval_method_loc = eval_lookup_table["pixel"](data_meta)



    # threshold = eval_lookup_table["pixel"](data_meta)
    det = eval_method_det.eval_auc()
    loc = eval_method_loc.eval_auc()

    all_fprs, all_pros = compute_pro(
        anomaly_maps=preds,
        ground_truth_maps=masks
    )
    aupro = trapezoid(all_fprs, all_pros)

    return det, loc, aupro
def performances1(fileinfos, preds, masks, config):
    ret_metrics = {}
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    for clsname in clsnames:
        preds_cls = []
        masks_cls = []
        for fileinfo, pred, mask in zip(fileinfos, preds, masks):
            if fileinfo["clsname"] == clsname:
                preds_cls.append(pred[None, ...])
                masks_cls.append(mask[None, ...])
        preds_cls = np.concatenate(np.asarray(preds_cls), axis=0)  # N x H x W
        masks_cls = np.concatenate(np.asarray(masks_cls), axis=0)  # N x H x W
        data_meta = EvalDataMeta(preds_cls, masks_cls)

        # auc
        if config.get("auc", None):
            for metric in config.auc:
                evalname = metric["name"]
                kwargs = metric.get("kwargs", {})
                eval_method = eval_lookup_table[evalname](data_meta, **kwargs)
                auc = eval_method.eval_auc()
                ret_metrics["{}_{}_auc".format(clsname, evalname)] = auc

    if config.get("auc", None):
        for metric in config.auc:
            evalname = metric["name"]
            evalvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for clsname in clsnames
            ]
            mean_auc = np.mean(np.array(evalvalues))
            ret_metrics["{}_{}_auc".format("mean", evalname)] = mean_auc

    return ret_metrics
# def performances(fileinfos, preds, masks, config):
#     ret_metrics = {}
#     clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
#
#     for clsname in clsnames:
#         # 收集当前类别的预测和真实标签
#         preds_cls = np.concatenate([pred[None, ...] for fileinfo, pred, mask in zip(fileinfos, preds, masks)
#                       if fileinfo["clsname"] == clsname], axis=0)
#         masks_cls = np.concatenate([mask[None, ...] for fileinfo, pred, mask in zip(fileinfos, preds, masks)
#                       if fileinfo["clsname"] == clsname], axis=0)
#
#         # 计算 AUC 类指标
#         if hasattr(config, 'auc'):
#             for metric in config.auc:
#                 evalname = metric["name"]
#                 kwargs = metric.get("kwargs", {})
#                 eval_method = eval_lookup_table[evalname](preds_cls, masks_cls, **kwargs)
#                 ret_metrics[f"{clsname}_{evalname}_auc"] = eval_method.eval_auc()
#
#         # 计算 AUPRO（新增）
#         if hasattr(config, 'aupro') and config.aupro:
#             ret_metrics[f"{clsname}_aupro"] = compute_aupro(preds_cls, masks_cls)
#
#     # 计算全局均值
#     for metric_type in ['auc', 'aupro']:
#         if hasattr(config, metric_type):
#             if metric_type == 'auc':
#                 for metric in config.auc:
#                     evalname = metric["name"]
#                     ret_metrics[f"mean_{evalname}_auc"] = np.mean(
#                         [ret_metrics[f"{clsname}_{evalname}_auc"] for clsname in clsnames]
#                     )
#             elif metric_type == 'aupro' and config.aupro:
#                 ret_metrics["mean_aupro"] = np.mean(
#                     [ret_metrics[f"{clsname}_aupro"] for clsname in clsnames]
#                 )
#
#     return ret_metrics

def log_metrics(ret_metrics, config):
    logger = logging.getLogger("global_logger")
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean"])) + ["mean"]

    # auc
    if config.get("auc", None):
        auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
        evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))
        record = Report(["clsname"] + evalnames)

        for clsname in clsnames:
            clsvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for evalname in evalnames
            ]
            record.add_one_record([clsname] + clsvalues)

        logger.info(f"\n{record}")


# def log_metrics(ret_metrics, config):
#     logger = logging.getLogger("global_logger")
#     clsnames = sorted(set(k.split("_")[0] for k in ret_metrics.keys()) - {"mean"}) + ["mean"]
#
#     # AUC 类指标表格
#     if hasattr(config, 'auc'):
#         auc_metrics = [m["name"] for m in config.auc]
#         auc_table = Report(["clsname"] + auc_metrics)
#         for clsname in clsnames:
#             row = [clsname]
#             for metric in auc_metrics:
#                 row.append(ret_metrics.get(f"{clsname}_{metric}_auc", "N/A"))
#             auc_table.add_one_record(row)
#         logger.info(f"AUC Metrics:\n{auc_table}")
#
#     # AUPRO 表格（新增）
#     if hasattr(config, 'aupro') and config.aupro:
#         aupro_table = Report(["clsname", "AUPRO"])
#         for clsname in clsnames:
#             aupro_table.add_one_record([clsname, ret_metrics.get(f"{clsname}_aupro", "N/A")])
#         logger.info(f"AUPRO Metrics:\n{aupro_table}")
