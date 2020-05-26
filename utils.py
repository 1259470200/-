import torch
import os
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from PIL import Image
import torchvision
import torch.nn as nn
from collections import OrderedDict
import time


def compute_dice(y_pred, y_true):
    """
    :param y_pred: 4-d tensor, value=[0, 1]
    :param y_true: 4-d tensor, value=[0, 1]
    :return:
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
    return np.sum(y_pred[y_true == 1])*2/(np.sum(y_pred)+np.sum(y_true))


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features*2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features*2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features*4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features*8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features*16, features*8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features*8)*2, features*8, name='dec4')

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name='dec3')

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name='dec2')

        self.upconv1 = nn.ConvTranspose2d(features * 2, features * 1, kernel_size=2, stride=2)
        self.decoder1 = UNet._block((features * 1) * 2, features * 1, name='dec1')

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + 'conv1',
                        nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
                    ),
                    (name + 'normal', nn.BatchNorm2d(num_features=features),),
                    (name + 'relu1', nn.ReLU(inplace=True),),
                    (
                        name + 'conv2',
                        nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
                    ),
                    (name + 'norma2', nn.BatchNorm2d(num_features=features),),
                    (name + 'relu2', nn.ReLU(inplace=True),),
                ]
            )
        )


class MyDataset(Dataset):
    # 需要自己写一个Dataset类，并且要继承从torch中import的Dataset基类，然后重写__len__和__getitem__两个方法，否则会报错
    # 此外还需要写__init__，传入数据所在路径和transform(用于数据预处理)
    def __init__(self, data_dir, transform=None):
        """
        :param data_dir: 读取的数据所在的路径
        :param transform: 数据预处理参数
        """
        self.data_info = self.dataInfo(data_dir)  # 用来读取数据信息(数据路径，标签)
        self.transform = transform

    def __getitem__(self, index):  # 根据索引读取数据路径再读取数据
        path_img, label_path = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        label = Image.open(label_path).convert('L')  # label是二值图像？不用转换吧，但是此处不转换后面的transforms不知道会不会出问题了

        # 单独对label做变换
        # label = self.transform.transforms[0](label)  # image做什么样的变换，那么label也做对应的变换
        # label = self.transform.transforms[-1](label)

        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)  # image做什么样的变换，那么label也做对应的变换
        else:  # 避免未作transforms而忘记把图
            # 像数据转化为tensor
            img = torch.tensor(img)
            label = torch.tensor(label)
        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def dataInfo(data_dir):  # 自定义函数用来获取数据信息，输入为数据所在路径，返回为一个元组(图像路径，标签路径)
        # 先读取所有的图像数据路径
        img_path = os.path.join(data_dir, 'images')
        imgs = os.listdir(img_path)
        # imgs.sort(key=lambda x: int(x.split('_')[0]))  # 根据图片标号从小到大排序
        label_path = os.path.join(data_dir, '1st_manual')
        labels = os.listdir(label_path)
        # labels.sort(key=lambda x: int(x.split('_')[0]))
        data_info = list()
        for i in range(len(imgs)):
            imgp = os.path.join(img_path, imgs[i])
            labelp = os.path.join(label_path, labels[i])
            data_info.append((imgp, labelp))
        return data_info


# 自定义transforms
class AddPepperNoise(object):
    def __init__(self, snr, p=0.9):
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        :param img: PIL image
        :return: PIL image
        """
        # 此处添加你对图像所做的一系列变换操作
        return img


class Evaluation(object):
    # reference: https://blog.csdn.net/sunflower_sara/article/details/81214897
    """
    AUC ＝ 1，代表完美分类器
    0.5 < AUC < 1，优于随机分类器
    0 < AUC < 0.5，差于随机分类器
    """
    def __init__(self, yt, yp):
        yt[yt != 0] = 1  # 因为有可能是255的，所以把非0的转化为1
        yp[yp != 0] = 1
        # 可以直接对矩阵做异或与或非运算
        self.TP = (yt & yp).sum()  # yt==1 and yp==1
        self.FP = (yp & (yt == 0)).sum()  # yp==1 and yt==0
        self.FN = ((yp == 0) & yt).sum()  # yp==0 and yt==1
        self.TN = ((yt == 0) & (yp == 0)).sum()  # yt==0 and yp==0

    # F1 score
    def F1(self):
        return 2*self.TP/(2*self.TP+self.FN+self.FP)

    def accuracy(self):
        return (self.TP+self.TN)/(self.TP+self.TN+self.FN+self.FP)

    def sensitivity(self):  # 召回率 / Recall / True positive rate / TPR / 灵敏度 /  敏感性 / sensitive/ 查全率
        return self.TP/(self.TP+self.FN)

    def specificity(self):  # false positive rate(FPR)，特异性
        return self.TN/(self.TN+self.FP)

    def ppv(self):  # precision/精确率/查准率
        return self.TP/(self.TP+self.FP)

    def npv(self):
        return self.TN/(self.TN+self.FN)


def transform_invert(img_, transform_train):  # 把tensor再转化为array
    """
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])
    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W->H*W*C
    # img_.detach().numpy()
    img_ = img_.detach().numpy()*255
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception('Invalid img shape, expected 1 or 3 in axis 2, but got {}'.format(img_.shape[2]))
    return img_


import pickle
def save_obj(obj, name):  # 保存dict
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):  # 加载dict
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    pass

# 使用示例
# index = {'a': 1, 'b': 2}
# save_obj(index, r'C:\Users\Administrator\Desktop\2020spring\MedicalImageAnalysis\results\index')
# index = load_obj(r'C:\Users\Administrator\Desktop\2020spring\MedicalImageAnalysis\models\index')
# print(np.mean(index['se']))
# print(np.mean(index['sp']))
# print(np.mean(index['acc']))
# print(np.mean(index['f1']))

# loss_dict = load_obj(r'C:\Users\Administrator\Desktop\2020spring\MedicalImageAnalysis\models\loss_dict')
# loss_dict = load_obj(r'C:\Users\Administrator\Desktop\2020spring\MedicalImageAnalysis\models\c2loss_dict')
# plt.figure()
# plt.subplot(221).plot(loss_dict['train_curve'][:1000]), plt.title('train_curve')
# plt.subplot(222).plot(loss_dict['valid_curve']), plt.title('valid_curve')
# plt.subplot(223).plot(loss_dict['train_dice_curve'][:1000]), plt.title('train_dice_curve')
# plt.subplot(224).plot(loss_dict['valid_dice_curve']), plt.title('valid_dice_curve')
# filename = r'C:\Users\Administrator\Desktop\2020spring\MedicalImageAnalysis\results\curve.png'
# plt.savefig(filename, bbox_inches='tight')
