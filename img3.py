from typing import Type, Union, List, Optional
import mindspore.nn as nn
from mindspore.common.initializer import Normal
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import dtype as mstype
from download import download
import numpy as np

data_dir = "d:/cifar-10-batches-bin"
batch_size = 256
image_size = 32
workers =4
num_classes =10

def create_dataset_cifar10(dataset_dir, usage, resize, batch_size, workers):
    data_set = ds.Cifar10Dataset(dataset_dir=dataset_dir,
#指定数据集的子集，可取值为‘train’、’test’或‘all’。取值为 ‘train’时将会读取50, 000个训练样本，取值为 ‘test’时将会读取10, 000 个测试样本，取值为 ‘all’时将会读取全部60, 000个样本。默认值：None，读取全部样本图片。
                                 usage=usage,
                                 num_parallel_workers=workers, #读取数据的工作线程数
                                 shuffle=True)          #是否混洗数据集
    trans = []
    if usage == "train":
        trans += [
            #对输入图像进行随机区域的裁剪。
            vision.RandomCrop((32, 32), (4, 4, 4, 4)),
            #对输入图像按给定的概率进行水平随机翻转。
            vision.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        vision.Resize(resize),
        #缩放和平移因子调整图片
        vision.Rescale(1.0 / 255.0, 0.0),
        #均值和标准差对输入图像（三维）进行归一化，output[channel] = (input[channel] - mean[channel]) / std[channel]
        vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        # <H, W, C> 转换为 <C, H, W>
        vision.HWC2CHW()
    ]
    #将输入的Tensor转换为指定的数据类型。
    target_trans = transforms.TypeCast(mstype.int32)

    # 数据映射操作
    data_set = data_set.map(operations=trans,
                            input_columns='image',
                            num_parallel_workers=workers)

    data_set = data_set.map(operations=target_trans,
                            input_columns='label',
                            num_parallel_workers=workers)
    # 批量操作
    data_set = data_set.batch(batch_size)

    return data_set


# 获取处理后的训练与测试数据集

dataset_train = create_dataset_cifar10(dataset_dir=data_dir,
                                       usage="train",
                                       resize=image_size,
                                       batch_size=batch_size,
                                       workers=workers)
step_size_train = dataset_train.get_dataset_size()

dataset_val = create_dataset_cifar10(dataset_dir=data_dir,
                                     usage="test",
                                     resize=image_size,
                                     batch_size=batch_size,
                                     workers=workers)
step_size_val = dataset_val.get_dataset_size()

# 初始化卷积层与BatchNorm的参数，生成一个服从正态分布N(sigma,mean)的随机数组用于初始化Tensor
weight_init = Normal(mean=0, sigma=0.02)
gamma_init = Normal(mean=1, sigma=0.02)

class ResidualBlockBase(nn.Cell):   # 继承nn.cell
    expansion: int = 1  # 最后一个卷积核数量与第一个卷积核数量相等

    def __init__(self, in_channel: int, out_channel: int,
                 stride: int = 1, norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None) -> None:
        super(ResidualBlockBase, self).__init__()  #调用nn.cell的初始化
        if not norm:
            self.norm = nn.BatchNorm2d(out_channel)
        else:
            self.norm = norm

        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=3, stride=stride,
                               weight_init=weight_init)
        self.conv2 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=3, weight_init=weight_init)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        """ResidualBlockBase construct."""
        identity = x  # shortcuts分支

        out = self.conv1(x)  # 主分支第一层：3*3卷积层
        out = self.norm(out)
        out = self.relu(out)
        out = self.conv2(out)  # 主分支第二层：3*3卷积层
        out = self.norm(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)
        out += identity  # 输出为主分支与shortcuts之和
        out = self.relu(out)

        return out
#
class ResidualBlock(nn.Cell):
    expansion = 4  # 最后一个卷积核的数量是第一个卷积核数量的4倍

    def __init__(self, in_channel: int, out_channel: int,
                 stride: int = 1, down_sample: Optional[nn.Cell] = None) -> None:
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=1, weight_init=weight_init)
        self.norm1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               kernel_size=3, stride=stride,
                               weight_init=weight_init)
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion,
                               kernel_size=1, weight_init=weight_init)
        self.norm3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):

        identity = x  # shortscuts分支

        out = self.conv1(x)  # 主分支第一层：1*1卷积层
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)  # 主分支第二层：3*3卷积层
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)  # 主分支第三层：1*1卷积层
        out = self.norm3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity     # 输出为主分支与shortcuts之和
        out = self.relu(out)

        return out

def make_layer(last_out_channel, block: Type[Union[ResidualBlockBase, ResidualBlock]],
               channel: int, block_nums: int, stride: int = 1):
    down_sample = None  # shortcuts分支

    if stride != 1 or last_out_channel != channel * block.expansion:
        #按照传入List的顺序依次将Cell添加
        down_sample = nn.SequentialCell([
            #https://zhuanlan.zhihu.com/p/353235794
            #转变原输入通道，使之能够与变化后的数据结合
            nn.Conv2d(last_out_channel, channel * block.expansion,
                      kernel_size=1, stride=stride, weight_init=weight_init),
            #https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.BatchNorm2d.html#mindspore.nn.BatchNorm2d
            nn.BatchNorm2d(channel * block.expansion, gamma_init=gamma_init)
        ])

    layers = []
    layers.append(block(last_out_channel, channel, stride=stride, down_sample=down_sample))

    in_channel = channel * block.expansion
    # 堆叠残差网络
    for _ in range(1, block_nums):

        layers.append(block(in_channel, channel))

    return nn.SequentialCell(layers)

from mindspore import load_checkpoint, load_param_into_net
#
#
class ResNet(nn.Cell):
    def __init__(self, block: Type[Union[ResidualBlockBase, ResidualBlock]],
                 layer_nums: List[int], num_classes: int, input_channel: int) -> None:
        super(ResNet, self).__init__()

        self.relu = nn.ReLU()
        # 第一个卷积层，输入channel为3（彩色图像），输出channel为64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, weight_init=weight_init)
        self.norm = nn.BatchNorm2d(64)
        # 最大池化层，缩小图片的尺寸
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        # 各个残差网络结构块定义
        self.layer1 = make_layer(64, block, 64, layer_nums[0])
        self.layer2 = make_layer(64 * block.expansion, block, 128, layer_nums[1], stride=2)
        self.layer3 = make_layer(128 * block.expansion, block, 256, layer_nums[2], stride=2)
        self.layer4 = make_layer(256 * block.expansion, block, 512, layer_nums[3], stride=2)
        # 平均池化层
        self.avg_pool = nn.AvgPool2d()
        # flattern层
        self.flatten = nn.Flatten()
        # 全连接层
        self.fc = nn.Dense(in_channels=input_channel, out_channels=num_classes)

    def construct(self, x):

        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x
#
#
def _resnet(model_url: str, block: Type[Union[ResidualBlockBase, ResidualBlock]],
            layers: List[int], num_classes: int, pretrained: bool, pretrained_ckpt: str,
            input_channel: int):
    model = ResNet(block, layers, num_classes, input_channel)

    if pretrained:
        # 加载预训练模型
        download(url=model_url, path=pretrained_ckpt, replace=True)
        param_dict = load_checkpoint(pretrained_ckpt)
        load_param_into_net(model, param_dict)

    return model
#
#
def resnet50(num_classes: int = 1000, pretrained: bool = False):
    "ResNet50模型"
    resnet50_url = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/models/application/resnet50_224_new.ckpt"
    resnet50_ckpt = "./LoadPretrainedModel/resnet50_224_new.ckpt"
    return _resnet(resnet50_url, ResidualBlock, [3, 4, 6, 3], num_classes,
                   pretrained, resnet50_ckpt, 2048)


# 定义ResNet50网络
network = resnet50(pretrained=True)
#
# # 全连接层输入层的大小
# in_channel = network.fc.in_channels
# fc = nn.Dense(in_channels=in_channel, out_channels=10)
# # 重置全连接层
# network.fc = fc
#
#
# # 设置学习率
# num_epochs = 5
# lr = nn.cosine_decay_lr(min_lr=0.00001, max_lr=0.001, total_step=step_size_train * num_epochs,
#                         step_per_epoch=step_size_train, decay_epoch=num_epochs)
# # 定义优化器和损失函数
# opt = nn.Momentum(params=network.trainable_params(), learning_rate=lr, momentum=0.9)
# loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
#
#
# def forward_fn(inputs, targets):
#     logits = network(inputs)
#     loss = loss_fn(logits, targets)
#     return loss
#
#
# grad_fn = ms.value_and_grad(forward_fn, None, opt.parameters)
#
#
# def train_step(inputs, targets):
#     loss, grads = grad_fn(inputs, targets)
#     opt(grads)
#     return loss
#
# import os
#
# # 创建迭代器
# data_loader_train = dataset_train.create_tuple_iterator(num_epochs=num_epochs)
# data_loader_val = dataset_val.create_tuple_iterator(num_epochs=num_epochs)
#
# # 最佳模型存储路径
# best_acc = 0
# best_ckpt_dir = "./BestCheckpoint"
# best_ckpt_path = "./BestCheckpoint/resnet50-best.ckpt"
#
# if not os.path.exists(best_ckpt_dir):
#     os.mkdir(best_ckpt_dir)
#
# import mindspore.ops as ops
#
#
# def train(data_loader, epoch):
#     """模型训练"""
#     losses = []
#     network.set_train(True)
#
#     for i, (images, labels) in enumerate(data_loader):
#         loss = train_step(images, labels)
#         if i % 100 == 0 or i == step_size_train - 1:
#             print('Epoch: [%3d/%3d], Steps: [%3d/%3d], Train Loss: [%5.3f]' %
#                   (epoch + 1, num_epochs, i + 1, step_size_train, loss))
#         losses.append(loss)
#
#     return sum(losses) / len(losses)
#
#
# def evaluate(data_loader):
#     """模型验证"""
#     network.set_train(False)
#
#     correct_num = 0.0  # 预测正确个数
#     total_num = 0.0  # 预测总数
#
#     for images, labels in data_loader:
#         logits = network(images)
#         pred = logits.argmax(axis=1)  # 预测结果
#         correct = ops.equal(pred, labels).reshape((-1, ))
#         correct_num += correct.sum().asnumpy()
#         total_num += correct.shape[0]
#
#     acc = correct_num / total_num  # 准确率
#
#     return acc
#
# # 开始循环训练
# print("Start Training Loop ...")
#
# for epoch in range(num_epochs):
#     curr_loss = train(data_loader_train, epoch)
#     curr_acc = evaluate(data_loader_val)
#
#     print("-" * 50)
#     print("Epoch: [%3d/%3d], Average Train Loss: [%5.3f], Accuracy: [%5.3f]" % (
#         epoch+1, num_epochs, curr_loss, curr_acc
#     ))
#     print("-" * 50)
#
#     # 保存当前预测准确率最高的模型
#     if curr_acc > best_acc:
#         best_acc = curr_acc
#         ms.save_checkpoint(network, best_ckpt_path)
#
# print("=" * 80)
# print(f"End of validation the best Accuracy is: {best_acc: 5.3f}, "
#       f"save the best ckpt file in {best_ckpt_path}", flush=True)

import matplotlib.pyplot as plt


def visualize_model(best_ckpt_path, dataset_val):
    num_class = 10  # 对狼和狗图像进行二分类
    net = resnet50(num_class)
    # 加载模型参数
    param_dict = ms.load_checkpoint(best_ckpt_path)
    ms.load_param_into_net(net, param_dict)
    # 加载验证集的数据进行验证
    data = next(dataset_val.create_dict_iterator())
    images = data["image"]
    labels = data["label"]
    # 预测图像类别
    output = net(data['image'])
    pred = np.argmax(output.asnumpy(), axis=1)

    # 图像分类
    classes = []

    with open(data_dir + "/batches.meta.txt", "r") as f:
        for line in f:
            line = line.rstrip()
            if line:
                classes.append(line)

    # 显示图像及图像的预测值
    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        # 若预测正确，显示为蓝色；若预测错误，显示为红色
        color = 'blue' if pred[i] == labels.asnumpy()[i] else 'red'
        plt.title('predict:{}'.format(classes[pred[i]]), color=color)
        picture_show = np.transpose(images.asnumpy()[i], (1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        picture_show = std * picture_show + mean
        picture_show = np.clip(picture_show, 0, 1)
        plt.imshow(picture_show)
        plt.axis('off')

    plt.show()


# 使用测试数据集进行验证
visualize_model(best_ckpt_path="./BestCheckpoint/resnet50-best.ckpt",dataset_val=dataset_val)