import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import models_lenet as models
import loader

from torchsummary import summary
import numpy as np

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--data', metavar='DIR', default='CUB_200_2011',
#                     help='path to dataset')
parser.add_argument('--initial_classes', default=5, type=int, metavar='N',
                    help='Number of MNIST classes for initial training. Choose from 0 to 8')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-c', '--checkpoint', default='pretrain_checkpoint_mnist', type=str, metavar='PATH',
                    help='path to save checkpoint (default: pretrain_checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate_mnist', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
    def __len__(self):
        return len(self.data)



def main():
    global args, best_prec1
    args = parser.parse_args()
    N_classes = args.initial_classes
    training_images_path = './mnist/training_images_npy/'
    training_labels_path = './mnist/training_labels_npy/'
    val_images_path = './mnist/testing_images_npy/'
    val_labels_path = './mnist/testing_labels_npy/'
    
    training_images = sorted(os.listdir(training_images_path))
    training_images = np.asarray(training_images)
    training_labels = sorted(os.listdir(training_labels_path))
    training_labels = np.asarray(training_labels)
    val_images = sorted(os.listdir(val_images_path))
    val_images = np.asarray(val_images)
    val_labels = sorted(os.listdir(val_labels_path))
    val_labels = np.asarray(val_labels)
    
    x_train = np.zeros((0, 28, 28))
    y_train = np.zeros((0))
    
    x_val = np.zeros((0, 28, 28))
    y_val = np.zeros((0))
    
    for i in range(0,N_classes):
        x = np.load(training_images_path+'/'+training_images[i])
        x_train = np.concatenate((x_train, x.astype(np.float)), axis =0)
        y = np.load(training_labels_path+'/'+training_labels[i])
        y_train = np.concatenate((y_train, y.astype(np.float)), axis =0)
#     x_train = x_train.transpose(0,3,2,1)
    x_train = np.expand_dims(x_train, axis=1)
    print('X_train Shape: {}'.format(x_train.shape))
    print('Y_train Shape: {}'.format(y_train.shape))


    for i in range(0,N_classes):
        xv = np.load(val_images_path+'/'+val_images[i])
        x_val = np.concatenate((x_val, xv.astype(np.float)), axis =0)
        yv = np.load(val_labels_path+'/'+val_labels[i])
        y_val = np.concatenate((y_val, yv.astype(np.float)), axis =0)
#     x_val = x_val.transpose(0,3,2,1)
    x_val = np.expand_dims(x_val, axis=1)
    print('X_val Shape: {}'.format(x_val.shape))
    print('Y_val Shape: {}'.format(y_val.shape))

    train_dataset = MyDataset(x_train, y_train)
    val_dataset = MyDataset(x_val, y_val)



    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    model = models.Net().cuda()
#     print(summary(model, (1, 28, 28)))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    extractor_params = list(map(id, model.extractor.parameters()))
    classifier_params = filter(lambda p: id(p) not in extractor_params, model.parameters())

    optimizer = torch.optim.SGD([
                {'params': model.extractor.parameters()},
                {'params': classifier_params, 'lr': args.lr * 10}
            ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # optionally resume from a checkpoint
    title = 'MNIST'
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Data loading code
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

#     train_dataset = loader.ImageLoader(
#         args.data,
#         transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ]), train=True)

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=args.batch_size, shuffle=True,
#         num_workers=args.workers, pin_memory=True)

#     val_loader = torch.utils.data.DataLoader(
#         loader.ImageLoader(args.data, transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             normalize,
#         ])),
#         batch_size=args.batch_size, shuffle=False,
#         num_workers=args.workers, pin_memory=True)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True)

#     if args.evaluate:
    validate(val_loader, model, criterion)
#         return

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        lr = optimizer.param_groups[1]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)

#         # evaluate on validation set
        test_loss, test_acc = validate(val_loader, model, criterion)

        # append logger file
#         logger.append([lr, train_loss, train_acc])
        logger.append([lr, train_loss, test_loss, train_acc, test_acc])


        # remember best prec@1 and save checkpoint
        is_best = test_acc > best_prec1
        best_prec1 = max(test_acc, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    bar = Bar('Training', max=len(train_loader))
    for batch_idx, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        model.weight_norm()
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    bar = Bar('Testing ', max=len(val_loader))
    with torch.no_grad():
        end = time.time()
        for batch_idx, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda()
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

             # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == '__main__':
    main()