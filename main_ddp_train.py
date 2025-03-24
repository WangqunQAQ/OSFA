import os
import csv
import time
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from utils import *
from models.model import OSFA

# 使用Accelerate的set_seed设置随机种子
seed = 42
set_seed(seed)


# 定义Top-k准确率计算函数
def accuracy(output, target, topk=(1, 5)):
    """计算Top-k准确率"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Early Stopping 类
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, mode="max"):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, metric):
        if self.mode == "max":
            if self.best_score is None or metric > self.best_score + self.delta:
                self.best_score = metric
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == "min":
            if self.best_score is None or metric < self.best_score - self.delta:
                self.best_score = metric
                self.counter = 0
            else:
                self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True


# 训练函数
def train_one_epoch(model, train_loader, criterion, optimizer, epoch, accelerator):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    model.train()
    running_loss = 0.0

    if accelerator.is_main_process:
        loader = tqdm(train_loader, desc="Training", leave=False)
    else:
        loader = train_loader

    end = time.time()
    for i, (images, labels) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        outputs = model(images)

        loss = 0
        weights = [1.0, 1.0, 1.0]
        for j, out in enumerate(outputs):
            loss += weights[j] * criterion(out, labels)

        # cls_loss = criterion(outputs[0], labels)
        # class_contrast_loss = criterion(outputs[1], labels)
        # scene_contrast_loss = criterion(outputs[2], labels)
        # loss = cls_loss + class_contrast_loss + scene_contrast_loss

        accelerator.backward(loss)
        optimizer.step()
        running_loss += loss.item()

        _outputs = accelerator.gather(outputs[0])
        _labels = accelerator.gather(labels)
        _loss = criterion(_outputs, _labels)

        acc1, acc5 = accuracy(_outputs, _labels, topk=(1, 5))
        losses.update(_loss.item(), _labels.size(0))
        top1.update(acc1.item(), _labels.size(0))
        top5.update(acc5.item(), _labels.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress._print(i)

    running_loss = accelerator.gather(torch.tensor(running_loss, device=accelerator.device)).mean().item()
    return running_loss


# 评估函数
def evaluate(model, test_loader, criterion, accelerator):

    model.eval()
    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    if accelerator.is_main_process:
        loader = tqdm(test_loader, desc="Evaluating", leave=False)
    else:
        loader = test_loader

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            outputs = model(images)

            outputs = accelerator.gather(outputs[0])
            labels = accelerator.gather(labels)

            if accelerator.num_processes > 1:
                mask = labels != -1
                outputs = outputs[mask]
                labels = labels[mask]

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            total_top1 += acc1.item() * labels.size(0)
            total_top5 += acc5.item() * labels.size(0)
            total_samples += labels.size(0)

    top1_acc = total_top1 / total_samples
    top5_acc = total_top5 / total_samples

    # assert total_samples == len(test_loader.dataset)
    return top1_acc, top5_acc


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='data/places365_standard/places365_standard',
                    help='path to dataset')
parser.add_argument('--model', metavar='MODEL', default='OSFA')
parser.add_argument('--base_name', metavar='BASE_NAME', default='resnet50_places365')
parser.add_argument('--clip_name', metavar='CLIP_NAME', default='ViT-B/32')
parser.add_argument('--load_pth', default='', type=str, metavar='PATH')
parser.add_argument('--base_dim', default=2048, type=int)
parser.add_argument('--clip_dim', default=512, type=int)
parser.add_argument('--hidden_dim', default=512, type=int)
parser.add_argument('--num_classes', default=365, type=int, help='num of class in the model')
parser.add_argument('--size_img', default=224, type=int, help='size of images')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--milestones', type=int, nargs='+', metavar='N')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--test_bs', default=365, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main():
    args = parser.parse_args()
    best_acc1 = 0.0

    save_dir = './work_dirs/' + args.model + '_' + args.base_name + '_' + args.clip_name.replace('-', '').replace('/', '') + '_' + args.data.split('/')[-1]
    os.makedirs(save_dir, exist_ok=True)

    # 设置日志
    logging.basicConfig(
        filename=os.path.join(save_dir, "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    # 模型、损失函数、优化器
    print("=> creating model '{}'".format(args.model))
    model = OSFA(base_dim=args.base_dim, clip_dim=args.clip_dim, hidden_dim=args.hidden_dim,
                 num_class=args.num_classes, base_name=args.base_name, clip_name=args.clip_name)

    if args.load_pth:
        if os.path.exists(args.load_pth):
            checkpoint = torch.load(args.load_pth)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['model_state_dict'].items()
                          if 'head' not in k}
            model.load_state_dict(state_dict, strict=False)
            for name, p in model.named_parameters():
                if 'backbone' in name:
                    p.requires_grad_(False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # 学习率调整策略
    milestones = args.milestones
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    start_epoch = args.start_epoch
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # 从上次保存的epoch继续
            best_acc1 = checkpoint['best_acc1']
            logging.info(f"Resuming training from epoch {start_epoch} with best accuracy {best_acc1:.2f}")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    if 'places' in args.data:
        traindir = os.path.join(args.data, 'train')
        testdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(testdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.test_bs, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif 'MITIndoor67' in args.data:
        traindir = os.path.join(args.data, 'train')
        testdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(testdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.test_bs, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif 'SUN397' in args.data:
        traindir = os.path.join(args.data, 'train')
        testdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(testdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.test_bs, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # 使用Accelerate初始化
    accelerator = Accelerator()
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

    if args.evaluate:
        acc1, acc5 = evaluate(model, test_loader, criterion, accelerator)
        print('acc1:', acc1, 'acc5:', acc5)
        return

    # 创建CSV文件
    csv_path = os.path.join(save_dir, "results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Train Loss", "Top-1 Acc", "Top-5 Acc"])

    # early_stopping = EarlyStopping(patience=5, delta=0.01, mode="max")

    for epoch in range(start_epoch, args.epochs):
        scheduler.step()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, accelerator)
        accelerator.wait_for_everyone()

        acc1, acc5 = evaluate(model, test_loader, criterion, accelerator)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            logging.info(f"Epoch {epoch+1}/{args.epochs}, "
                         f"Loss: {train_loss:.4f}, "
                         f"Top-1 Acc: {acc1:.2f}%, "
                         f"Top-5 Acc: {acc5:.2f}%")
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch + 1, train_loss, acc1, acc5])

            if acc1 > best_acc1:
                best_acc1 = acc1
                save_path = os.path.join(save_dir, f"{args.model}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc1': best_acc1,
                }, save_path)
                logging.info(f"Best model saved to {save_path} with Top-1 Acc: {best_acc1:.2f}")

            # early_stopping(acc1)
            # if early_stopping.early_stop:
            #     logging.info("Early stopping triggered. Training stopped.")
            #     break


if __name__ == '__main__':
    main()

