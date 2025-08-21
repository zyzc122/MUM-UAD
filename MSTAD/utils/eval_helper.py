import glob
import logging
import os

import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from sklearn import metrics
from easydict import EasyDict as edict
from .metrics import compute_pro, trapezoid


def dump(save_dir, outputs):
    filenames = outputs["filename"]
    batch_size = len(filenames)
    pred_score = outputs["pred_score"]  # B x 1 x H x W
    # pred_score = outputs["pred_score"].cpu().numpy()  # B x 1 x H x W
    img_has_anomalys = outputs["img_has_anomaly"].cpu().numpy()  # B x 1 x H x W
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
            score=pred_score[i],
            img_has_anomalys=img_has_anomalys[i],
            pred=preds[i],
            mask=masks[i],
            height=heights[i],
            width=widths[i],
            clsname=clsnames[i],
        )


def merge_together(save_dir):
    npz_file_list = glob.glob(os.path.join(save_dir, "*.npz"))
    fileinfos = []
    preds = []
    masks = []
    scores = []
    img_has_anomalys = []
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
        scores.append(npz["score"])
        img_has_anomalys.append(npz["img_has_anomalys"])

    preds = np.concatenate(np.asarray(preds), axis=0)  # N x H x W
    masks = np.concatenate(np.asarray(masks), axis=0)  # N x H x W
    scores = np.asarray(scores)
    # scores = np.concatenate(np.asarray(scores), axis=0)  # N x H x W
    img_has_anomalys = np.concatenate(np.asarray(img_has_anomalys), axis=0)  # N x H x W
    return fileinfos, preds, masks, img_has_anomalys, scores


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
        masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(np.int)  # (N, )
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

class EvalPerPixelThreshold:
    def __init__(self, data_meta):
        # 全部拉平 224*224*图片数量
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        # 计算阈值
        threshold = numpy.percentile(self.preds, 99.8)
        return threshold


class EvalPerPixelAUC:
    def __init__(self, data_meta):
        # 全部拉平 224*224*图片数量
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
    "threshold": EvalPerPixelThreshold,
}

object_list = ['cable', 'zipper', 'capsule', 'transistor', 'bottle', 'hazelnut',
               'metal_nut',  'toothbrush', 'pill', 'screw']
texture_list = ['leather', 'carpet',  'tile', 'grid', 'wood']


def uni_performances(fileinfos, preds, masks, config, img_has_anomalys, scores):
    # ret_metrics = {}
    # clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    # for clsname in clsnames:
    # preds_cls = []
    # masks_cls = []
    # for fileinfo, pred, mask in zip(fileinfos, preds, masks):
        # 取出test中 某一类的全部图片
        # if fileinfo["clsname"] == clsname:
        # preds_cls.append(pred[None, ...])
        # masks_cls.append(mask[None, ...])
    # preds_cls = np.concatenate(np.asarray(preds), axis=0)  # N x H x W
    # masks_cls = np.concatenate(np.asarray(masks), axis=0)  # N x H x W
    fpr, tpr, thresholds = metrics.roc_curve(img_has_anomalys, scores, pos_label=1)
    det = metrics.auc(fpr, tpr)
    # 存储异常得分

    mask_ = (img_has_anomalys==0)
    scores_good = scores[mask_]
    mask_ = (img_has_anomalys == 1)
    scores_defe = scores[mask_]
    # np.save(
    #     "/home/zyz-4090/PycharmProjects/Saliency_MSTAD_from3090/Saliency_MSTAD/tools/vis_density/density_ndarray_MUMAD_sigmoid/preds_good",
    #     scores_good)
    # np.save(
    #     "/home/smart-solution-server-003/anomaly_PycharmProjects/UniAD_Gradient/tools/vis_density/density_ndarray_MUMAD_sigmoid/preds_defe",
    #     scores_defe)
    data_meta = EvalDataMeta(preds, masks)
    eval_method_loc = eval_lookup_table["pixel"](data_meta)
    loc = eval_method_loc.eval_auc()

    all_fprs, all_pros = compute_pro(
        anomaly_maps=preds,
        ground_truth_maps=masks
    )
    aupro = trapezoid(all_fprs, all_pros)

    return loc, det, aupro

def performances(fileinfos, preds, masks, config):
    ret_metrics = {}
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    for clsname in clsnames:
        preds_cls = []
        masks_cls = []
        for fileinfo, pred, mask in zip(fileinfos, preds, masks):
            # 取出test中 某一类的全部图片
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
            object_scores = []
            texture_scores = []
            evalname = metric["name"]
            for clsname in clsnames:
                if clsname in object_list:
                    evalvalues = ret_metrics["{}_{}_auc".format(clsname, evalname)]
                    object_scores.append(evalvalues)
                else:
                    evalvalues = ret_metrics["{}_{}_auc".format(clsname, evalname)]
                    texture_scores.append(evalvalues)
            objectmean_auc = np.mean(np.array(object_scores))
            texturemean_auc = np.mean(np.array(texture_scores))
            ret_metrics["{}_{}_{}_auc".format("object", "mean", evalname)] = objectmean_auc
            ret_metrics["{}_{}_{}_auc".format("texture", "mean", evalname)] = texturemean_auc

            evalvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for clsname in clsnames
            ]
            mean_auc = np.mean(np.array(evalvalues))
            ret_metrics["{}_{}_auc".format("mean", evalname)] = mean_auc
            # evalvalues = [
            #     ret_metrics["{}_{}_auc".format(clsname, evalname)]
            #     for clsname in clsnames
            # ]
        # for metric in config.auc:
        #     evalname = metric["name"]
        #     evalvalues = [
        #         ret_metrics["{}_{}_auc".format(clsname, evalname)]
        #         for clsname in clsnames
        #     ]
        #     mean_auc = np.mean(np.array(evalvalues))
        #     ret_metrics["{}_{}_auc".format("mean", evalname)] = mean_auc
    return ret_metrics

def M_distance(fileinfos, preds, masks, config):
    ret_metrics = {}
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    for clsname in clsnames:
        preds_cls = []
        masks_cls = []
        for fileinfo, pred, mask in zip(fileinfos, preds, masks):
            # 取出test中 某一类的全部图片
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



def log_metrics(ret_metrics, config):
    logger = logging.getLogger("global_logger")
    clsnames = list(set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()]))
    clsnames.sort()
    clsnames.remove('mean')
    clsnames.remove('object_mean')
    clsnames.remove('texture_mean')

    clsnames.append('object_mean')
    clsnames.append('texture_mean')
    clsnames.append('mean')
    # auc
    if config.get("auc", None):
        auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
        evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))
        evalnames.sort()
        record = Report(["clsname"] + evalnames)

        for clsname in clsnames:
            clsvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for evalname in evalnames
            ]
            record.add_one_record([clsname] + clsvalues)

        logger.info(f"\n{record}")
