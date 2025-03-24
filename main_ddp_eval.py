import os
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.model import OSFA


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
parser.add_argument('--test_bs', default=365, type=int)


# 定义Top-k准确率计算函数
def accuracy(output, target, topk=(1, 5)):
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


def evaluate_10crop(model, test_loader, device):
    model.eval()
    total_top1l = 0
    total_top5l = 0
    total_top1s = 0
    total_top5s = 0
    total_top1c = 0
    total_top5c = 0
    total_samples = 0

    loader = tqdm(test_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)  # 将输入数据移动到设备上
            labels = labels.to(device)  # 将标签数据移动到设备上

            batch_size, num_crops, c, h, w = images.size()
            images = images.view(batch_size * num_crops, c, h, w)

            outputs = model(images)

            out0 = outputs[0].view(batch_size, num_crops, -1).mean(dim=1)
            out1 = outputs[2].view(batch_size, num_crops, -1).mean(dim=1)
            out2 = outputs[1].view(batch_size, num_crops, -1).mean(dim=1)

            acc1l, acc5l = accuracy(out0, labels, topk=(1, 5))
            total_top1l += acc1l.item() * labels.size(0)
            total_top5l += acc5l.item() * labels.size(0)

            acc1s, acc5s = accuracy(out1, labels, topk=(1, 5))
            total_top1s += acc1s.item() * labels.size(0)
            total_top5s += acc5s.item() * labels.size(0)

            acc1c, acc5c = accuracy(out2, labels, topk=(1, 5))
            total_top1c += acc1c.item() * labels.size(0)
            total_top5c += acc5c.item() * labels.size(0)

            total_samples += labels.size(0)

    top1l_acc = total_top1l / total_samples
    top5l_acc = total_top5l / total_samples

    top1s_acc = total_top1s / total_samples
    top5s_acc = total_top5s / total_samples

    top1c_acc = total_top1c / total_samples
    top5c_acc = total_top5c / total_samples
    return [top1l_acc, top5l_acc, top1s_acc, top5s_acc, top1c_acc, top5c_acc]


def evaluate_1crop(model, test_loader, device):
    model.eval()
    total_top1l = 0
    total_top5l = 0
    total_top1s = 0
    total_top5s = 0
    total_top1c = 0
    total_top5c = 0
    total_samples = 0

    loader = tqdm(test_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)  # 将输入数据移动到设备上
            labels = labels.to(device)  # 将标签数据移动到设备上

            outputs = model(images)

            acc1l, acc5l = accuracy(outputs[0], labels, topk=(1, 5))
            total_top1l += acc1l.item() * labels.size(0)
            total_top5l += acc5l.item() * labels.size(0)

            acc1s, acc5s = accuracy(outputs[2], labels, topk=(1, 5))
            total_top1s += acc1s.item() * labels.size(0)
            total_top5s += acc5s.item() * labels.size(0)

            acc1c, acc5c = accuracy(outputs[1], labels, topk=(1, 5))
            total_top1c += acc1c.item() * labels.size(0)
            total_top5c += acc5c.item() * labels.size(0)

            total_samples += labels.size(0)

    top1l_acc = total_top1l / total_samples
    top5l_acc = total_top5l / total_samples

    top1s_acc = total_top1s / total_samples
    top5s_acc = total_top5s / total_samples

    top1c_acc = total_top1c / total_samples
    top5c_acc = total_top5c / total_samples
    return [top1l_acc, top5l_acc, top1s_acc, top5s_acc, top1c_acc, top5c_acc]


def main():
    args = parser.parse_args()

    # 模型、损失函数、优化器
    print("=> creating model '{}'".format(args.model))
    model = OSFA(base_dim=args.base_dim, clip_dim=args.clip_dim, hidden_dim=args.hidden_dim,
                 num_class=args.num_classes, base_name=args.base_name, clip_name=args.clip_name)

    testdir = os.path.join(args.data, 'val')
    test_loader_1crop = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])),
        batch_size=args.test_bs, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader_10crop = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])),
        batch_size=args.test_bs, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # 初始化评估设备（只使用单个GPU或CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.load_pth:
        if os.path.exists(args.load_pth):
            checkpoint = torch.load(args.load_pth)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
            model.load_state_dict(state_dict, strict=True)
    # 将模型移动到设备上（单个GPU或CPU）
    model = model.to(device)

    # 执行评估
    top1l_acc, top5l_acc, top1s_acc, top5s_acc, top1c_acc, top5c_acc = evaluate_1crop(model, test_loader_1crop, device)
    # 输出结果
    print("1-crop result:")
    print(f"linear Top-1 Accuracy: {top1l_acc:.2f}%")
    print(f"linear Top-5 Accuracy: {top5l_acc:.2f}%")
    print(f"scene Top-1 Accuracy: {top1s_acc:.2f}%")
    print(f"scene Top-5 Accuracy: {top5s_acc:.2f}%")
    print(f"class Top-1 Accuracy: {top1c_acc:.2f}%")
    print(f"class Top-5 Accuracy: {top5c_acc:.2f}%")

    # 执行评估
    top1l_acc, top5l_acc, top1s_acc, top5s_acc, top1c_acc, top5c_acc = evaluate_10crop(model, test_loader_10crop, device)
    # 输出结果
    print("10-crop result:")
    print(f"linear Top-1 Accuracy: {top1l_acc:.2f}%")
    print(f"linear Top-5 Accuracy: {top5l_acc:.2f}%")
    print(f"scene Top-1 Accuracy: {top1s_acc:.2f}%")
    print(f"scene Top-5 Accuracy: {top5s_acc:.2f}%")
    print(f"class Top-1 Accuracy: {top1c_acc:.2f}%")
    print(f"class Top-5 Accuracy: {top5c_acc:.2f}%")


if __name__ == '__main__':
    main()
