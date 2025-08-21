import glob
import logging
import os
from matplotlib import colors
import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import matplotlib
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import matplotlib.font_manager as font_manager


def dump(save_dir, outputs):
    filenames = outputs["filename"]
    batch_size = len(filenames)
    preds = outputs["pred"].cpu().numpy()  # B x 1 x H x W
    masks = outputs["mask"].cpu().numpy()  # B x 1 x H x W
    heights = outputs["height"].cpu().numpy()
    widths = outputs["width"].cpu().numpy()
    attn_output_weights = outputs["attn_output_weights"].cpu().numpy()
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
            attn_output_weights=attn_output_weights[i]
        )


def merge_together(save_dir):
    npz_file_list = glob.glob(os.path.join(save_dir, "*.npz"))

    attn_output_weights = []
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
        attn_output_weights.append(npz["attn_output_weights"])

    preds = np.concatenate(np.asarray(preds), axis=0)  # N x H x W
    masks = np.concatenate(np.asarray(masks), axis=0)  # N x H x W
    # attn_output_weights = np.concatenate(np.asarray(attn_output_weights), axis=0)  # N x H x W
    return fileinfos, preds, masks, attn_output_weights


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
btad_name = [
    '01',
    '02',
    '03',
]


def performances(fileinfos, preds, masks, config, attn_output_weights):
    ret_metrics = {}
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    plt.rcParams['font.size'] = 22
    # plt.rcParams['font.sans-serif']=['Times New Roman']

    #这句话rebuild是特别重要，不然会提示找不到文件
    # font_manager._rebuild()
    footpath='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    prop=font_manager.FontProperties(fname=footpath)
    matplotlib.rcParams['font.family']=prop.get_name()
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.fonttype'] = 'path'

    fig = plt.figure(figsize=(25, 9))
    count = 0
    color_list = []
    for clsname in all_names:
        preds_cls = []
        masks_cls = []
        attn_output_weights_cls = []
        for fileinfo, pred, mask, attn_output_weight in zip(fileinfos, preds, masks, attn_output_weights):
            # 取出test中 某一类的全部图片
            if fileinfo["clsname"] == clsname:
                preds_cls.append(pred[None, ...])
                masks_cls.append(mask[None, ...])
                attn_output_weights_cls.append(attn_output_weight[None, ...])

        # 选择了指定类别后 输出可视化图像
        arr = np.array(attn_output_weights_cls)
        arr_min = np.min(arr, axis=-1, keepdims=True)
        arr_max = np.max(arr, axis=-1, keepdims=True)
        normal_data = (arr - arr_min) / (arr_max - arr_min)
        normal_data = np.squeeze(normal_data)
        # mean_arr = np.sum(arr, axis=0)
        # normal_data = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        ax = fig.add_subplot(5, 3, count+1)
        # ax.text(family='Times New Roman')
        color_list.append(ax)
        count = count + 1
        # if count != 15:
        #     ax.xaxis.set_ticks([])
        # else:
        #     ax.tick_params(axis='x', labelsize = 30)
        ax.tick_params(axis='x', labelsize = 22)
        x1_label = ax.get_xticklabels()
        [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
        ax.yaxis.set_ticks([])
        ax.imshow(normal_data, cmap='jet', aspect="auto")
        ax.set_title("{}".format(clsname), fontproperties='Times New Roman',
                     fontsize=22, loc='center', pad=8)
        # ax.set_ylabel("{}".format(clsname), fontsize=35, rotation=0, x=-0.1, y=0.35)
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

    dataset1 = np.random.randint(0, 50, size=(10, 10))
    dataset2 = np.random.randint(50, 100, size=(10, 10))
    vmin = 0.01 * min(np.min(dataset1), np.min(dataset2))
    vmax = 0.01 * max(np.max(dataset1), np.max(dataset2))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    a = ax.pcolormesh(dataset1, norm=norm, cmap=plt.get_cmap('jet'))
    # plt.show()

    plt.tight_layout(pad=1, h_pad=1, w_pad=1)
    # plt.subplots_adjust(hspace=0)
    fig.colorbar(a, ax=color_list, shrink=0.8)
    plt.savefig('/home/scu-its-gpu-001/UniAD_Gradient/tools/11111111111.png')

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
