import glob
import logging
import os

import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

def dump(save_dir, outputs):
    filenames = outputs["filename"]
    batch_size = len(filenames)
    preds = outputs["pred"].cpu().numpy()  # B x 1 x H x W
    masks = outputs["mask"].cpu().numpy()  # B x 1 x H x W
    heights = outputs["height"].cpu().numpy()
    widths = outputs["width"].cpu().numpy()
    src = outputs["src"]
    output_decoder = outputs["output_decoder"]
    clsnames = outputs["clsname"]
    for i in range(batch_size):
        file_dir, filename = os.path.split(filenames[i])
        _, subname = os.path.split(file_dir)
        filename = "{}_{}_{}".format(clsnames[i], subname, filename)
        filename, _ = os.path.splitext(filename)
        save_file = os.path.join(save_dir, filename + ".npz")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.savez(
            save_file,
            filename=filenames[i],
            pred=preds[i],
            mask=masks[i],
            height=heights[i],
            width=widths[i],
            clsname=clsnames[i],
            src=src[i],
            output_decoder=output_decoder[i]
        )


def merge_together(save_dir):
    npz_file_list = glob.glob(os.path.join(save_dir, "*.npz"))

    src = []
    output_decoder = []
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
        src.append(npz["src"])
        output_decoder.append(npz["output_decoder"])

    preds = np.concatenate(np.asarray(preds), axis=0)  # N x H x W
    masks = np.concatenate(np.asarray(masks), axis=0)  # N x H x W
    # attn_output_weights = np.concatenate(np.asarray(attn_output_weights), axis=0)  # N x H x W
    return fileinfos, preds, masks, src, output_decoder


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
}

object_list = ['cable', 'zipper', 'capsule', 'transistor', 'bottle', 'hazelnut',
               'metal_nut',  'toothbrush', 'pill', 'screw']
texture_list = ['leather', 'carpet',  'tile', 'grid', 'wood']

all_names = ['carpet', 'grid', 'leather', 'tile', 'wood',
             'bottle', 'cable', 'capsule', 'hazelnut','metal_nut','pill', 'screw','toothbrush', 'transistor',
             'zipper']


def performances(fileinfos_good, preds_good, masks_good,
                 fileinfos_bad, preds_bad, masks_bad,
                 config,
                 src_good, output_decoder_good, src_bad, output_decoder_bad):
    ret_metrics = {}
    # clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    fig = plt.figure(figsize=(50, 50))
    count = 0
    new_count = 0
    for clsname in all_names:
        preds_cls = []
        masks_cls = []
        src_cls = []
        output_decoder_cls = []
        for fileinfo, pred, mask, src_, output_decoder_ in zip(fileinfos_good, preds_good, masks_good,
                                                               src_good, output_decoder_good):
            if fileinfo["clsname"] == clsname:
                preds_cls.append(pred[None, ...])
                masks_cls.append(mask[None, ...])
                src_cls.append(src_[None, ...])
                output_decoder_cls.append(output_decoder_[None, ...])

        src_cls_arr = np.array(src_cls)
        src_data = src_cls_arr
        # src_data = (src_cls_arr - np.min(src_cls_arr)) / (np.max(src_cls_arr) - np.min(src_cls_arr))
        output_decoder_cls_arr = np.array(output_decoder_cls)
        output_decoder_data = output_decoder_cls_arr
        # output_decoder_data = (output_decoder_cls_arr - np.min(output_decoder_cls_arr)) / \
        #            (np.max(output_decoder_cls_arr) - np.min(output_decoder_cls_arr))
        ax = fig.add_subplot(5, 3, count+1)
        count = count + 1
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_title(clsname, fontweight ="bold", loc='center')
        # plt.figure(count)
        for i in range(src_data.shape[0]):
            ax.scatter([src_data[i][0][0], output_decoder_data[i][0][0]],
                       [src_data[i][0][1], output_decoder_data[i][0][1]],
                        c=['red', 'blue'], s=15, marker='o')
        # ax.legend(
        #     (ax.scatter(src_data[0][0][0], src_data[0][0][1],
        #                  c='red', s=20, marker='o'),
        #      ax.scatter(output_decoder_data[0][0][0], output_decoder_data[0][0][1],
        #                  c='blue', s=20, marker='o')),
        #     ('input_good', 'output_good'),
        #     loc='upper right'
        # )

        preds_cls_new = []
        masks_cls_new = []
        src_cls_new = []
        output_decoder_cls_new = []
        for fileinfo, pred, mask, src_, output_decoder_ in zip(fileinfos_bad, preds_bad, masks_bad,
                                                               src_bad, output_decoder_bad):
            if fileinfo["clsname"] == clsname:
                preds_cls_new.append(pred[None, ...])
                masks_cls_new.append(mask[None, ...])
                src_cls_new.append(src_[None, ...])
                output_decoder_cls_new.append(output_decoder_[None, ...])

        src_cls_arr = np.array(src_cls_new)
        src_data = src_cls_arr
        # src_data = (src_cls_arr - np.min(src_cls_arr)) / (np.max(src_cls_arr) - np.min(src_cls_arr))
        output_decoder_cls_arr = np.array(output_decoder_cls_new)
        output_decoder_data = output_decoder_cls_arr
        # output_decoder_data = (output_decoder_cls_arr - np.min(output_decoder_cls_arr)) / \
        #                       (np.max(output_decoder_cls_arr) - np.min(output_decoder_cls_arr))
        ax = plt.subplot(5, 3, new_count+1)
        new_count = new_count + 1
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        # plt.figure(count)
        for i in range(src_data.shape[0]):
            ax.scatter([src_data[i][0][0], output_decoder_data[i][0][0]],
                       [src_data[i][0][1], output_decoder_data[i][0][1]],
                       c=['black', 'green'], s=15, marker='o')
        ax.legend(
            (ax.scatter(src_data[0][0][0], src_data[0][0][1],
                         c='red', s=20, marker='o'),
             ax.scatter(output_decoder_data[0][0][0], output_decoder_data[0][0][1],
                         c='blue', s=20, marker='o'),
             ax.scatter(src_data[0][0][0], src_data[0][0][1],
                         c='black', s=20, marker='o'),
              ax.scatter(output_decoder_data[0][0][0], output_decoder_data[0][0][1],
                         c='green', s=20, marker='o')
             ),
            ('input_good', 'output_good', 'input_anomaly', 'output_anomaly'),
            loc='upper right'
        )

    plt.savefig('/home/scu-its-gpu-001/UniAD_Gradient/tools/maps_after_project.png')

    # if config.get("auc", None):
    #     for metric in config.auc:
    #         object_scores = []
    #         texture_scores = []
    #         evalname = metric["name"]
    #         for clsname in clsnames:
    #             if clsname in object_list:
    #                 evalvalues = ret_metrics["{}_{}_auc".format(clsname, evalname)]
    #                 object_scores.append(evalvalues)
    #             else:
    #                 evalvalues = ret_metrics["{}_{}_auc".format(clsname, evalname)]
    #                 texture_scores.append(evalvalues)
    #         objectmean_auc = np.mean(np.array(object_scores))
    #         texturemean_auc = np.mean(np.array(texture_scores))
    #         ret_metrics["{}_{}_{}_auc".format("object", "mean", evalname)] = objectmean_auc
    #         ret_metrics["{}_{}_{}_auc".format("texture", "mean", evalname)] = texturemean_auc
    #
    #         evalvalues = [
    #             ret_metrics["{}_{}_auc".format(clsname, evalname)]
    #             for clsname in clsnames
    #         ]
    #         mean_auc = np.mean(np.array(evalvalues))
    #         ret_metrics["{}_{}_auc".format("mean", evalname)] = mean_auc
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

def performances_single(fileinfos_good, preds_good, masks_good,
                 fileinfos_bad, preds_bad, masks_bad,
                 config,
                 src_good, output_decoder_good, src_bad, output_decoder_bad):
    ret_metrics = {}
    # clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    fig = plt.figure(figsize=(8, 8))
    count = 0
    new_count = 0
    preds_cls = []
    masks_cls = []
    src_cls = []
    output_decoder_cls = []
    for fileinfo, pred, mask, src_, output_decoder_ in zip(fileinfos_good, preds_good, masks_good,
                                                           src_good, output_decoder_good):
        preds_cls.append(pred[None, ...])
        masks_cls.append(mask[None, ...])
        src_cls.append(src_)
        # src_cls.append(src_[None, ...])
        output_decoder_cls.append(output_decoder_)

    src_cls_good_arr = np.array(src_cls)
    # min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))#默认为范围0~1，拷贝操作
    # src_good_shift = src_cls_arr - np.min(src_cls_arr)
    # src_data = min_max_scaler.fit_transform(src_shift)
    # src_data = src_cls_arr
    # src_data = (src_cls_arr - np.min(src_cls_arr)) / (np.max(src_cls_arr) - np.min(src_cls_arr))

    output_decoder_good_cls_arr = np.array(output_decoder_cls)
    # output_decoder_good_shift = src_cls_arr - np.min(output_decoder_cls_arr)
    # output_decoder_data = min_max_scaler.fit_transform(output_decoder_shift)
    # output_decoder_data = output_decoder_cls_arr
    # output_decoder_data = (output_decoder_cls_arr - np.min(output_decoder_cls_arr)) / \
    #            (np.max(output_decoder_cls_arr) - np.min(output_decoder_cls_arr))
    # ax = fig.add_subplot(5, 3, count+1)
    # count = count + 1

    # ax.xaxis.set_ticks([])
    # ax.yaxis.set_ticks([])
    # ax.set_title(fileinfo["clsname"], fontweight ="bold", loc='center')
    # plt.figure(count)

    # ax.legend(
    #     (ax.scatter(src_data[0][0][0], src_data[0][0][1],
    #                  c='red', s=20, marker='o'),
    #      ax.scatter(output_decoder_data[0][0][0], output_decoder_data[0][0][1],
    #                  c='blue', s=20, marker='o')),
    #     ('input_good', 'output_good'),
    #     loc='upper right'
    # )

    preds_cls_new = []
    masks_cls_new = []
    src_cls_new = []
    output_decoder_cls_new = []
    for fileinfo, pred, mask, src_, output_decoder_ in zip(fileinfos_bad, preds_bad, masks_bad,
                                                           src_bad, output_decoder_bad):
        preds_cls_new.append(pred[None, ...])
        masks_cls_new.append(mask[None, ...])
        src_cls_new.append(src_)
        output_decoder_cls_new.append(output_decoder_)

    src_cls_bad_arr = np.array(src_cls_new)
    # src_data = src_cls_arr
    # src_bad_shift = src_cls_arr - np.min(src_cls_arr)
    # src_data = min_max_scaler.fit_transform(src_shift)
    # src_data = (src_cls_arr - np.min(src_cls_arr)) / (np.max(src_cls_arr) - np.min(src_cls_arr))

    output_decoder_bad_cls_arr = np.array(output_decoder_cls_new)
    # output_decoder_bad_shift = src_cls_arr - np.min(output_decoder_cls_arr)
    # output_decoder_data = min_max_scaler.fit_transform(output_decoder_shift)
    # output_decoder_data = output_decoder_cls_arr
    # output_decoder_data = (output_decoder_cls_arr - np.min(output_decoder_cls_arr)) / \
    #                       (np.max(output_decoder_cls_arr) - np.min(output_decoder_cls_arr))

    all_cls = np.concatenate([src_cls_good_arr, output_decoder_good_cls_arr,
                              src_cls_bad_arr, output_decoder_bad_cls_arr], axis=0)
    min_ = np.min(all_cls)
    max_ = np.max(all_cls)
    src_good_data = (src_cls_good_arr - min_) / (max_ - min_)
    # src_good_data = src_cls_good_arr - min_
    # src_good_data = min_max_scaler.fit_transform(src_cls_good_arr)

    # output_decoder_good_data = output_decoder_good_cls_arr - min_
    output_decoder_good_data = (output_decoder_good_cls_arr - min_) / (max_ - min_)
    # output_decoder_good_data = min_max_scaler.fit_transform(output_decoder_good_cls_arr)

    # src_cls_bad_data = src_cls_bad_arr - min_
    src_cls_bad_data = (src_cls_bad_arr - min_) / (max_ - min_)
    # src_cls_bad_data = min_max_scaler.fit_transform(src_cls_bad_arr)

    output_decoder_bad_data = (output_decoder_bad_cls_arr - min_) / (max_ - min_)
    # output_decoder_bad_data = output_decoder_bad_cls_arr - min_
    # output_decoder_bad_data = min_max_scaler.fit_transform(output_decoder_bad_cls_arr)


    # plt.xticks([])
    # plt.yticks([])
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title(fileinfo["clsname"], fontweight ="bold", loc='center')

    for i in range(src_good_data.shape[0]):
        plt.scatter([src_good_data[i][0], output_decoder_good_data[i][0]],
                    [src_good_data[i][1], output_decoder_good_data[i][1]],
                    c=['red', 'blue'], s=3, marker='o')

    for i in range(src_cls_bad_data.shape[0]):
        plt.scatter([src_cls_bad_data[i][0], output_decoder_bad_data[i][0]],
                   [src_cls_bad_data[i][1], output_decoder_bad_data[i][1]],
                   c=['black', 'green'], s=3, marker='o')
    plt.legend(
        (plt.scatter(src_good_data[0][0], src_good_data[0][1],
                    c='red', s=20, marker='o'),
         plt.scatter(output_decoder_good_data[0][0], output_decoder_good_data[0][1],
                    c='blue', s=20, marker='o'),
         plt.scatter(src_cls_bad_data[0][0], src_cls_bad_data[0][1],
                    c='black', s=20, marker='o'),
         plt.scatter(output_decoder_bad_data[0][0], output_decoder_bad_data[0][1],
                    c='green', s=20, marker='o')
         ),
        ('input_good', 'rec_good', 'input_anomaly', 'rec_anomaly'),
        bbox_to_anchor=(0.75, 1),
        ncol=2,
        loc='lower left'
    )
    plt.tight_layout()

    plt.savefig('/home/scu-its-gpu-001/UniAD_Gradient/tools/distance_{}.png'.format(fileinfo["clsname"]))

    # if config.get("auc", None):
    #     for metric in config.auc:
    #         object_scores = []
    #         texture_scores = []
    #         evalname = metric["name"]
    #         for clsname in clsnames:
    #             if clsname in object_list:
    #                 evalvalues = ret_metrics["{}_{}_auc".format(clsname, evalname)]
    #                 object_scores.append(evalvalues)
    #             else:
    #                 evalvalues = ret_metrics["{}_{}_auc".format(clsname, evalname)]
    #                 texture_scores.append(evalvalues)
    #         objectmean_auc = np.mean(np.array(object_scores))
    #         texturemean_auc = np.mean(np.array(texture_scores))
    #         ret_metrics["{}_{}_{}_auc".format("object", "mean", evalname)] = objectmean_auc
    #         ret_metrics["{}_{}_{}_auc".format("texture", "mean", evalname)] = texturemean_auc
    #
    #         evalvalues = [
    #             ret_metrics["{}_{}_auc".format(clsname, evalname)]
    #             for clsname in clsnames
    #         ]
    #         mean_auc = np.mean(np.array(evalvalues))
    #         ret_metrics["{}_{}_auc".format("mean", evalname)] = mean_auc
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
