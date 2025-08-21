import cv2
import numpy
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import os
from os.path import join
import numpy as np
from torchvision import transforms
import os
import os
import random
from glob import glob
import kornia

def MVTec_Path_Split(train_ratio, root_image,
                     category, random_seed=10):
    path = f'/{root_image}/{category}/test/'
    if os.path.exists(path):
        defect_name = os.listdir(path)
    # defect_name = os.listdir(glob(os.path.join(root_image, category, "test"))[0])
    # defect_name.remove('good')
    defect_name.sort()
    train_image_files = []
    val_image_files = []

    for dename in defect_name:
        # self.train_temps = glob(os.path.join(root_image, category, "test", dename, "*.png"))
        # 随机打乱 并 按比例划分
        files = glob(f'/{root_image}/{category}/test/{dename}/*.JPG')
        files.extend(glob(f'/{root_image}/{category}/test/{dename}/*.png'))
        files.extend(glob(f'/{root_image}/{category}/test/{dename}/*.bmp'))
        train_images, val_images = random_split_list(files, train_ratio, random_seed)
        train_image_files.extend(train_images)
        val_image_files.extend(val_images)
    return train_image_files, val_image_files

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root_image, root_refine, pred_methods,
                 category, input_size, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.root_refine = root_refine
        self.category = category
        self.pred_methods = pred_methods
        self.defect_name = os.listdir(glob(os.path.join(root_image, category, "test"))[0])
        # self.defect_name.remove('good')
        self.defect_name.sort()
        self.train_image_files = []
        self.val_image_files = []
        self.label_refine_files = []
        if is_train:
            for dename in self.defect_name[:1]:
                self.train_image_files.extend(glob(os.path.join(root_image, category, "test",
                                                                dename, "*.png")))
                # self.train_image_files = glob(os.path.join(root_image, category, "test", dename, "*.png"))
                self.train_image_files.sort()
            self.image_files = self.train_image_files

            # self.image_files = glob(os.path.join(root_image, category, "test", self.defect_name[0], "*.png"))
            # self.image_files.sort()
        else:
            for dename in self.defect_name[1:]:
                self.val_image_files.extend(glob(os.path.join(root_image, category, "test", dename, "*.png")))
            self.val_image_files.sort()
            self.image_files = self.val_image_files

        self.target_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ]
        )
        self.pred_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
            ]
        )
        self.is_train = is_train

    def __getitem__(self, index):
        if self.is_train:
            image_file = self.image_files[index]
            image = Image.open(image_file).convert('RGB')
            image = self.image_transform(image)
            target = Image.open(
                image_file.replace("/test/", "/ground_truth/").replace(
                    ".png", "_mask.png"
                )
            )
            target = self.target_transform(target)
            preds_list = []
            # dename = image_file.split("/")[7]
            for method in self.pred_methods:
                for dename in self.defect_name[:1]:
                    self.label_refine_files.extend(glob(os.path.join(self.root_refine, method,
                                                                self.category, dename, "*.npy")))
                self.label_refine_files.sort()
                pred = numpy.load(self.label_refine_files[index])
                pred = self.normalize(pred)
                pred_tensor = torch.from_numpy(pred).float()
                pred_tensor = self.pred_transform(pred_tensor)
                preds_list.append(pred_tensor)
            batch = {'image': image,
                     'gt': target,
                     'flabels': torch.cat(preds_list, 0)
                    }
        else:
            image_file = self.image_files[index]
            image = Image.open(image_file).convert('RGB')
            image = self.image_transform(image)
            target = Image.open(
                image_file.replace("/test/", "/ground_truth/").replace(
                    ".png", "_mask.png"
                )
            )
            target = self.target_transform(target)
            # for method in self.pred_methods:
            #     for dename in self.defect_name[1:]:
            #         self.val_image_files.extend(glob(os.path.join(self.root_refine,self. category,
            #                                                       method, dename, "*.npy")))
            # self.val_image_files.sort()
            batch = {'image': image,
                     'gt': target,}

        return batch

    def __len__(self):
        return len(self.image_files)

    def normalize(self, pred, max_value=None, min_value=None):
        if max_value is None or min_value is None:
            return (pred - pred.min()) / (pred.max() - pred.min())
        else:
            return (pred - min_value) / (max_value - min_value)

class MVTecDataset_all(torch.utils.data.Dataset):
    def __init__(self, ratio, root_image, root_refine, pred_methods,
                 category, input_size, is_train=True, random_seed = 10):
        random.seed(random_seed)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.root_image = root_image
        self.root_refine = root_refine
        self.category = category
        self.pred_methods = pred_methods
        self.defect_name = os.listdir(glob(os.path.join(root_image,
                                                        category, "test"))[0])
        # self.defect_name.remove('good')
        self.defect_name.sort()
        self.train_image_files = []
        self.val_image_files = []
        self.label_refine_files = []
        self.ratio = ratio
        for dename in self.defect_name:
            # self.train_temps = glob(os.path.join(root_image, category, "test", dename, "*.png"))
            # 随机打乱 并 按比例划分
            train_images, val_images = random_split_list(glob(os.path.join(root_image, category,
                                                                           "test", dename, "*.png")),
                              self.ratio)
            self.train_image_files.extend(train_images)
            self.val_image_files.extend(val_images)
        self.target_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ]
        )
        self.pred_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                # transforms.ToTensor(),
            ]
        )
        self.is_train = is_train

    def __getitem__(self, index):
        input = {}
        if self.is_train:
            image_file = self.train_image_files[index]
            image = Image.open(image_file).convert('RGB')
            image = self.image_transform(image)
            input.update(
                {
                    "filename": image_file,
                    "height": image.shape[1],
                    "width": image.shape[2],
                    "clsname": self.category,
                    "image": image,
                }
            )
            target = Image.open(
                image_file.replace("/test/", "/ground_truth/").replace(
                    ".png", "_mask.png"
                )
            )
            target = self.target_transform(target)
            input.update({'mask': target})
            dename = image_file.split("/")[8]
            preds_list = []
            for method in self.pred_methods:
                # self.label_refine_files.extend()
                flabel_path = os.path.join(self.root_refine, method, self.category, dename,
                                           os.path.split(image_file)[1].split('.')[0]) + '.npy'
                pred = numpy.load(flabel_path)
                pred = self.normalize(pred)
                pred_tensor = torch.from_numpy(pred).float()
                if method == 'cflow':
                    pred_tensor = pred_tensor.unsqueeze(0)
                pred_tensor = self.pred_transform(pred_tensor)
                preds_list.append(pred_tensor)
            input.update({
                'flabels': torch.cat(preds_list, 0)
            })
        else:
            image_file = self.val_image_files[index]
            image = Image.open(image_file).convert('RGB')
            image = self.image_transform(image)
            target = Image.open(
                image_file.replace("/test/", "/ground_truth/").replace(
                    ".png", "_mask.png"
                )
            )
            target = self.target_transform(target)
            input.update(
                {
                    "filename": image_file,
                    "height": image.shape[1],
                    "width": image.shape[2],
                    "clsname": self.category,
                    "image": image,
                    'mask': target
                }
            )

        return input

    def __len__(self):
        return len(self.val_image_files)

    def normalize(self, pred, max_value=None, min_value=None):
        if max_value is None or min_value is None:
            return (pred - pred.min()) / (pred.max() - pred.min())
        else:
            return (pred - min_value) / (max_value - min_value)

class MVTecDataset_val(torch.utils.data.Dataset):
    def __init__(self, root, input_size, category):

        self.category = category
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.root = root
        self.target_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        input = {}
        image_file = self.root[index]
        image = Image.open(image_file).convert('RGB')
        image = self.image_transform(image)
        target = Image.open(
            image_file.replace("/test/", "/ground_truth/").replace(
                ".png", "_mask.png"
            )
        )
        target = self.target_transform(target)
        input.update(
            {
                "filename": image_file,
                "height": image.shape[1],
                "width": image.shape[2],
                "clsname": self.category,
                "image": image,
                'mask': target
            }
        )
        return input

    def __len__(self):
        return len(self.root)


def random_split_list(input, ratio, random_seed):

    input.sort()
    random.seed(random_seed)
    random.shuffle(input)

    index = int(len(input) * ratio)
    # index = 6

    list1 = input[:index]
    list2 = input[index:]

    return list1, list2

class MVTecDataset_ratio(torch.utils.data.Dataset):
    def __init__(self, root_image, methods, root_flabels=None,
                 input_size=224, is_train=True, random_seed=10):
        self.root_image = root_image
        self.is_train = is_train
        self.root_flabels = root_flabels
        self.methods = methods
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.target_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
            ]
        )
        self.pred_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
            ]
        )

    def __getitem__(self, index):
        input = {}

        image_file = self.root_image[index]
        image = Image.open(image_file).convert('RGB')

        # Y = (numpy.array(Image.open(image_file).convert('L').resize((224,224), Image.ANTIALIAS))>127).astype(numpy.float64)
        # kernel = numpy.ones((5,5))
        # C = cv2.dilate(Y, kernel)
        # cv2.imwrite('aa1.png', C*255)
        # C = cv2.erode(Y, kernel)
        # cv2.imwrite('bb1.png', C*255)
        # C = cv2.dilate(Y, kernel) - cv2.erode(Y, kernel)
        # cv2.imwrite('cc1.png', C*255)
        # gray = cv2.cvtColor(C, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('3333.png', gray)
        # image_ = cv2.imread(image_file, cv2.IMREAD_COLOR)
        # image_ = cv2.resize(image_, dsize=(224, 224))
        # image_ = image_ / 255.0
        # image_ = image_.astype(np.float32)
        # imagegray = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
        # imagegray = imagegray[None, :, :,]
        # imagegray = imagegray[:, None, :, :]
        # torchvision.utils.save_image(gradient, '311.png')
        # imagegray = np.transpose(imagegray, (2, 0, 1))

        image = self.image_transform(image)
        if image_file.split('/')[-2] == 'good':
            target = np.zeros((image.shape[1], image.shape[2])).astype(np.uint8)
            target = Image.fromarray(target, "L")
            img_has_anomaly = np.array([0], dtype=np.float32)
        else:
            # target_path = image_file.replace("/test/", "/ground_truth/").replace(".JPG", "_mask.png")
            target_path = image_file.replace("/test/", "/ground_truth/").split('.')[0]
            if os.path.exists(target_path + '.png'):
                target_path = target_path + '.png'
            elif os.path.exists(target_path + '.bmp'):
                target_path = target_path + '.bmp'
            target = Image.open(target_path)
            img_has_anomaly = np.array([1], dtype=np.float32)
        target = self.target_transform(target)
        category = image_file.split('/')[-4]
        input.update(
            {
                "filename": image_file,
                "height": image.shape[1],
                "width": image.shape[2],
                "clsname": category,
                "image": image,
                'mask': target,
                'img_has_anomaly': img_has_anomaly,
                # 'gradient': torch.Tensor(imagegray)
            }
        )
        if self.is_train:
            dename = image_file.split("/")[-2]
            preds_list = []
            flabels_has_anomaly_list = []
            for method in self.methods:
                # self.label_refine_files.extend()
                file_name = os.path.split(image_file)[1].split('.')[0]
                flabel_path_jpg = f'/{self.root_flabels}/{method}/{category}/{dename}/{file_name}.JPG'
                flabel_path_png = f'/{self.root_flabels}/{method}/{category}/{dename}/{file_name}.png'
                if os.path.exists(flabel_path_jpg):
                    flabel_path = flabel_path_jpg
                elif os.path.exists(flabel_path_png):
                    flabel_path = flabel_path_png
                pred = numpy.array(Image.open(flabel_path).convert('L'))
                # pred = numpy.load(flabel_path)
                # pred = self.normalize(pred)
                pred_tensor = torch.from_numpy(pred).float().clone()
                pred_tensor = pred_tensor.unsqueeze(0)
                # if method == 'cflow':
                #     pred_tensor = pred_tensor.unsqueeze(0)
                pred_tensor = self.pred_transform(pred_tensor)
                preds_list.append(pred_tensor)
                flabels_has_anomaly_list.append(img_has_anomaly)
            input.update({
                'flabels': torch.cat(preds_list, 0),
                'flabels_has_anomaly': numpy.concatenate(flabels_has_anomaly_list, 0)
            })

        return input

    def __len__(self):
        return len(self.root_image)

    def normalize(self, pred, max_value=None, min_value=None):
        if max_value is None or min_value is None:
            return (pred - pred.min()) / (pred.max() - pred.min())
        else:
            return (pred - min_value) / (max_value - min_value)
