import argparse
import os
import shutil
import time
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

#model_names = sorted(name for name in models.__dict__
#    if name.islower() and not name.startswith("__")
#    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#parser.add_argument('data', metavar='DIR',
#                    help='path to dataset')
#parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
 #                   choices=model_names,
 #                   help='model architecture: ' +
 #                       ' | '.join(model_names) +
 #                       ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=74, type=int, metavar='N',
                    help='number of total epochs to run MNISTdefault = 74')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=50, type=int,
                    metavar='N', help='mini-batch size (MNISTdefault: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5*1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--no-cuda', action='store_true', default = False, help ='false')

best_prec1 = 0
vgg16_config = [64,64, 'Max', 128, 128, 'Max', 256, 256, 256, 'Max', 512, 512, 512, 'Max',512, 512, 512, 'Max']

class vggNet(nn.Module):
    def __init__(self,conv_features, fc_features):
        super(vggNet, self).__init__()
#        self.conv_layers = make_conv_layers(vgg16_config)
#        self.fc_layers = make_fc_layers()
        self.conv_layers = conv_features
        self.fc_layers = fc_features
        init_weight(self)


    def forward(self,x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def init_weight(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0]* m.kernel_size[1]*m.out_channels
            m.weight.data.normal_(0, math.sqrt(2/n))
            m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            m.weight.data.normal_(0, 0.02)



def make_conv_layers(cfg):
    layers = []
    in_channels = 3
    for ii in cfg:
        if ii == 'Max':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, ii, kernel_size=3, padding=1)
            layers += [conv2d,nn.BatchNorm2d(ii), nn.ReLU(inplace=True)]
            in_channels = ii
    return nn.Sequential(*layers)


def make_fc_layers():
    return nn.Sequential(
            nn.Linear(512, 4096), 
            nn.ReLU(True), 
            nn.Dropout(), 
            nn.Linear(4096, 4096), 
            nn.ReLU(True), 
            nn.Dropout(), 
            nn.Linear(4096,10)
            )

def main():
    
#SAVE filename
    path_current = os.path.dirname(os.path.realpath(__file__))
    path_subdir = 'dataset'
    data_filename = 'VGGdataset.txt'
    
    path_file = os.path.join(path_current,path_subdir,data_filename)
    f=open(path_file,'w')
    



    global args, best_prec1
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    # create model
#    if args.pretrained:
#        print("=> using pre-trained model '{}'".format(args.arch))
#        model = models.__dict__[args.arch](pretrained=True)
#    else:
#        print("=> creating model '{}'".format(args.arch))
#        model = models.__dict__[args.arch]()
#
#    if not args.distributed:
#        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
#            model.features = torch.nn.DataParallel(model.features)
#            model.cuda()
#        else:
#            model = torch.nn.DataParallel(model).cuda()
#    else:
#        model.cuda()
#        model = torch.nn.parallel.DistributedDataParallel(model)
#
    device = torch.device("cuda" if use_cuda else "cpu")
#    model = vggNet().cuda().to("cuda")
    model = vggNet(make_conv_layers(vgg16_config),make_fc_layers()).cuda()

    def init_vgg(net):
        import math
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                c_out, _, kh, kw = m.weight.size()
                n = kh * kw * c_out
                m.weight.data.normal_(0, math.sqrt(2 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    init_vgg(model)

# define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
#    traindir = os.path.join(args.data, 'train')
#    valdir = os.path.join(args.data, 'val')
#    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
#
#    train_dataset = datasets.ImageFolder(
#        traindir,
#        transforms.Compose([
#            transforms.RandomResizedCrop(224),
#            transforms.RandomHorizontalFlip(),
#            transforms.ToTensor(),
#            normalize,
#        ]))
#
#
#
#    if args.distributed:
#        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
#    else:
#        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            '../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32,4),
                #               transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

#    train_loader = torch.utils.data.DataLoader(
#        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
#        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                '../data',
                train=False,
                download=True,
                transform = transforms.Compose([
#                    transforms.RandomHorizontalFlip(),
#                    transforms.ColorJitter(), 
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        mean = [0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]
                     ),
                ])
            ),
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.workers, 
            pin_memory=True
        )

#    val_loader = torch.utils.data.DataLoader(
#        datasets.ImageFolder(valdir, transforms.Compose([
#            transforms.Resize(256),
#            transforms.CenterCrop(224),
#            transforms.ToTensor(),
#            normalize,
#        ])),
#        batch_size=args.batch_size, shuffle=False,
#        num_workers=args.workers, pin_memory=True)

#    if args.evaluate:
#        validate(val_loader, model, criterion)
#        return





    train_losses =np.zeros((args.epochs))
    train_prec1s =np.zeros((args.epochs))
    eval_losses =np.zeros((args.epochs))
    eval_prec1s =np.zeros((args.epochs))
    x_epoch = np.zeros((args.epochs))
#    x_epoch = np.linspace(args.start_epoch,args.epochs -1, args.epochs,endpoint=True)

    
    
    
    
    
    
    
    
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss, train_prec1 = train(train_loader, model, criterion, optimizer, epoch,f)

        # evaluate on validation set
        eval_loss, eval_prec1 = validate(val_loader, model, criterion,f)



        train_losses[epoch] = train_loss 
        train_prec1s[epoch] = train_prec1
        eval_losses[epoch] = eval_loss
        eval_prec1s[epoch] = eval_prec1
        x_epoch[epoch] = epoch
#        train_losses =np.append( train_losses + train_loss
#        train_prec1s = train_prec1s + train_prec1
#        eval_losses = eval_losses + eval_loss
#        eval_prec1s = eval_prec1s + eval_prec1
##
#        train_loss.append(train_losses)
#        train_prec1.append(train_prec1s)
#        eval_loss.append(eval_losses)
#        eval_prec1.append(eval_prec1s)









        # remember best prec@1 and save checkpoint
        is_best = eval_prec1 > best_prec1
        best_prec1 = max(eval_prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'VGG16',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)










    matplotlib.use('Agg')
    
    plt.clf()
    plt.close()
    fig_loss = plt.figure()
    ax_loss = fig_loss.add_subplot(1,1,1)
    ax_loss.plot(x_epoch,train_losses,label='Train Loss')
    ax_loss.plot(x_epoch,eval_losses,label='Test Loss')
    ax_loss.legend(loc=1)
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.set_title('Loss of Train and Test')
    plot_loss_filename = 'VGGloss.png'
    path_loss_file = os.path.join(path_current,path_subdir,plot_loss_filename)
    fig_loss.savefig(path_loss_file)

    plt.clf()
    plt.close()
    fig_prec = plt.figure()
    ax_prec = fig_prec.add_subplot(1,1,1)
    ax_prec.plot(x_epoch,train_prec1s,label='Train Best1')
    ax_prec.plot(x_epoch,eval_prec1s,label='Test Best1')
    ax_prec.legend(loc=1)
    ax_prec.set_xlabel('epoch')
    ax_prec.set_ylabel('loss')
    ax_prec.set_title('Best1 of Train and Test')
    plot_prec_filename = 'VGGprec.png'
    path_prec_file = os.path.join(path_current,path_subdir,plot_prec_filename)
    fig_prec.savefig(path_prec_file)


    f.close()







def train(train_loader, model, criterion, optimizer, epoch,f):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        img = img.cuda()
        # compute output
        output = model(img)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1[0], img.size(0))
        top5.update(prec5[0], img.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            f.write('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.2f}({top1.avg:.2f})\t'
                  'Prec@5 {top5.val:.2f}({top5.avg:.2f})\r\n'.format(
                   epoch, i, len(train_loader),
                   loss=losses, top1=top1, top5=top5))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                   epoch, i, len(train_loader),
                   loss=losses, top1=top1, top5=top5))
        
    return losses.avg, top1.avg


def validate(val_loader, model, criterion,f):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (img, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            img = img.cuda()

            # compute output
            output = model(img)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1[0], img.size(0))
            top5.update(prec5[0], img.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                f.write('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\r\n'.format(
                       i, len(val_loader), loss=losses,
                       top1=top1, top5=top5))
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), loss=losses,
                       top1=top1, top5=top5))

        f.write(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\r\n'
              .format(top1=top1, top5=top5))
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
