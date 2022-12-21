from __future__ import print_function
import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from cifar10 import get_cifar10
from wideresnet import build_wideresnet
from utils import _get_logger, Cosine_LR_Scheduler
from copy import deepcopy
from tqdm import tqdm


def update_ema_variables(model, model_teacher, alpha):
    # Use the true average until the exponential average is more correct
    # alpha = min(1.0 - 1.0 / float(global_step + 1), alpha)
    for param_t, param in zip(model_teacher.parameters(), model.parameters()):
        param_t.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    return model, model_teacher


def consistency_loss(logits_w1, logits_w2):
    logits_w2 = logits_w2.detach()
    assert logits_w1.size() == logits_w2.size()
    return F.mse_loss(torch.softmax(logits_w1, dim=-1), torch.softmax(logits_w2, dim=-1), reduction='mean')


def train(model, ema_model, label_loader, unlabel_loader, test_loader, args):
    model.train()
    ema_model.train()
    label_iter = iter(label_loader)
    cur_itrs = 0
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    acc = 0.0
    ema_acc = 0.0

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay)

    lr_scheduler = Cosine_LR_Scheduler(optimizer, warmup_epochs=10, num_epochs=args.epochs, base_lr=args.lr,
                                       iter_per_epoch=args.step_size)

    while True:

        for idx, ((img1_ul, img2_ul), target_no_label) in enumerate(tqdm(unlabel_loader)):
            cur_itrs += 1
            try:
                (img1, img2), target_label = next(label_iter)
            except StopIteration:
                label_iter = iter(label_loader)
                (img1, img2), target_label, = next(label_iter)

            batch_size_labeled = img1.shape[0]
            input1 = Variable(torch.cat([img1, img1_ul]).to(args.device))
            input2 = Variable(torch.cat([img2, img2_ul]).to(args.device))
            target = Variable(target_label.to(args.device))

            optimizer.zero_grad()
            output = model(input1)

            # forward pass with mean teacher
            # torch.no_grad() prevents gradients from being passed into mean teacher model
            with torch.no_grad():
                ema_output = ema_model(input2)

            unsup_loss = consistency_loss(output, ema_output)
            out_x = output[:batch_size_labeled]
            sup_loss = ce_loss(out_x, target)
            warm_up = float(np.clip((cur_itrs) / (args.unsup_warm_up * args.total_itrs), 0., 1.))
            loss = sup_loss + warm_up * args.lambda_u * unsup_loss  # 损失为 有标签的交叉熵损失+ 一致性损失(基于平滑性假设，一个模型对于 一个输入及其变形应该保持一致性）
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            args.writer.add_scalar('train/loss', loss.item(), cur_itrs)
            args.writer.add_scalar('train/lr', lr, cur_itrs)
            args.writer.add_scalar('train/warm_up', warm_up, cur_itrs)
            model, ema_model = update_ema_variables(model, ema_model, args.ema_m)  # 更新模型

            if cur_itrs % args.step_size == 0:
                args.logger.info(
                    "Train cur_itrs: [{}/{} ({:.0f}%)]\t loss:{:.6f}\t warm_up:{:.6f}".format(
                        cur_itrs,
                        args.total_itrs,
                        100. * cur_itrs / args.total_itrs,
                        loss.item(),
                        warm_up,
                    ))
                tmp_acc = test(model, test_loader=test_loader, args=args, )
                args.writer.add_scalar('acc', tmp_acc, cur_itrs)
                if tmp_acc > acc:
                    acc = tmp_acc
                    #  保存模型
                    torch.save({
                        "cur_itrs": cur_itrs,
                        "model": model,
                        "ema_model": ema_model,
                        "optimizer": optimizer,
                        "best_acc": acc
                    }, args.model_save_path)
                tmp_acc = test(ema_model, test_loader=test_loader, args=args)
                args.writer.add_scalar('ema_acc', tmp_acc, cur_itrs)
                if tmp_acc > ema_acc:
                    ema_acc = tmp_acc

            if cur_itrs > args.total_itrs:
                return


def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    args.logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))

    model.train()
    return acc


def main():
    args = EasyDict()
    args.total_itrs = 2 ** 20
    args.step_size = 5000  # 每 step_size 步骤 保存一次数据
    args.epochs = args.total_itrs // args.step_size + 1
    args.label = 4000  # 使用有标签的数量
    args.lambda_u = 50
    args.unsup_warm_up = 0.4
    args.ema_m = 0.999
    args.datasets = "cifar10"
    args.num_classes = 10
    args.seed = 0  # 随机种子
    args.batch_size = 64  # 有标签的batchsize
    args.K = 3  # 无标签的batchsize是 有标签的K 倍
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    args.lr = 0.03
    args.opt = "sgd"
    args.weight_decay = 0.0005
    args.momentum = 0.9

    args.datapath = "./data"
    args.save_path = "checkpoint/12-19-mean-teacher"
    root = os.path.dirname(os.path.realpath(__file__))
    args.save_path = os.path.join(root, args.save_path)
    args.model_save_path = os.path.join(args.save_path, "mean_teacher_model.pth")

    args.writer = SummaryWriter(args.save_path)
    args.logger = _get_logger(os.path.join(args.save_path, "log.log"), "info")

    model = build_wideresnet(widen_factor=3, depth=28, dropout=0.1, num_classes=args.num_classes).to(args.device)
    ema_model = deepcopy(model)

    # model=nn.DataParallel()
    #  设置ema_model不计算梯度
    for name, param in ema_model.named_parameters():
        param.requires_grad = False

    # 设置随机种子。
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 50000个样本，只使用4000label
    # mean-teacher
    # 4000label acc=90%
    # 2000label acc=87%
    # 1000label acc=83%
    # 500 label acc=58%
    # 250 label acc=52%
    label_loader, unlabel_loader, test_loader = get_cifar10(root=args.datapath, n_labeled=args.label,
                                                            batch_size=args.batch_size, K=args.K)
    args.logger.info("label_loader length: {}".format(len(label_loader) * args.batch_size))
    args.logger.info("unlabel_loader length: {}".format(len(unlabel_loader) * args.batch_size * args.K))
    args.logger.info("test_loader length: {}".format(len(test_loader)))
    args.logger.info("length: {}".format(args.total_itrs // len(unlabel_loader)))

    train(model=model, ema_model=ema_model,
          label_loader=label_loader, unlabel_loader=unlabel_loader, test_loader=test_loader, args=args)

    # mean_teacher = build_wideresnet(widen_factor=3, depth=28, dropout=0.1, num_classes=10).to(device)
    # optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=arg))
    # lr_scheduler, num_epochs = create_scheduler(arg, optimizer)

    # for epoch in range(num_epochs):
    #     train(model, mean_teacher, device, label_loader, unlabel_loader, lr_scheduler, optimizer, epoch, num_epochs,
    #           logger)
    #     test(model, device, test_loader, logger)
    #     test(mean_teacher, device, test_loader, logger)
    #     # 保存模型
    #     torch.save(model, "mean_teacher_cifar10.pt")


if __name__ == '__main__':
    main()
