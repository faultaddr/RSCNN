import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
from torchvision import transforms
from models import RSCNN_MSN_Seg as RSCNN_MSN
from data import ShapeNetPart
import utils.pytorch_utils as pt_utils
import data.data_utils as d_utils
import argparse
import random
import yaml
import time
from context_prior import get_loss
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='Relation-Shape CNN Shape Part Segmentation Training')
parser.add_argument('--config', default='cfgs/config_msn_partseg.yaml', type=str)

def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:'%(k), v)
    print("\n**************************\n")

    try:
        os.makedirs(args.save_path)
    except OSError:
        pass

    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])

    train_dataset = ShapeNetPart(root = args.data_root, num_points = args.num_points, split = 'trainval', normalize = True, transforms = train_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=True
    )

    global test_dataset
    test_dataset = ShapeNetPart(root = args.data_root, num_points = args.num_points, split = 'test', normalize = True, transforms = test_transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=True
    )

    model = RSCNN_MSN(num_classes = args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)

    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))

    #criterion = nn.CrossEntropyLoss()
    criterion=get_loss()
    num_batch = len(train_dataset)/args.batch_size

    # training
    train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch):
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()   # initialize augmentation
    global Class_mIoU, Inst_mIoU
    Class_mIoU, Inst_mIoU = 0.83, 0.85
    batch_count = 0
    #model.train()
    for epoch in range(args.epochs):
        model.train()
        losses=[]
        start_time=time.time()
        for i, data in enumerate(train_dataloader):

            points, target, cls = data
            target = target.cuda()
            points = points.cuda()
            one_hot_target=to_categorical(target,50)
            print('target',target.shape)
            # augmentation
            points = PointcloudScaleAndTranslate(points)
            optimizer.zero_grad()

            batch_one_hot_cls = np.zeros((len(cls), 16))   # 16 object classes
            for b in range(len(cls)):
                batch_one_hot_cls[b, int(cls[b])] = 1
            batch_one_hot_cls = torch.from_numpy(batch_one_hot_cls).float().cuda()
            # batch_one_hot_cls = Variable(batch_one_hot_cls.float().cuda())

            pred,context = model(points, batch_one_hot_cls)
            pred = pred.view(-1, args.num_classes)
            target = target.view(-1,1)[:,0]
            loss = criterion(pred, target,None,context,one_hot_target)

            loss.backward()

            optimizer.step()
            losses.append(loss.item())
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if bnm_scheduler is not None:
                bnm_scheduler.step(epoch-1)
            #if i % args.print_freq_iter == 0:
            #    print('[epoch %3d: %3d/%3d] \t train loss: %0.6f \t lr: %0.5f' %(epoch+1, i, num_batch, loss.data.clone(), lr_scheduler.get_lr()[0]))
            batch_count += 1

            # validation in between an epoch
            #if (epoch < 3 or epoch > 40) and args.evaluate and batch_count % int(args.val_freq_epoch * num_batch) == 0:
        end_time=time.time()
        print('[epoch %3d time=%d s] \t train loss: %0.6f \t lr: %0.5f' %(epoch+1, end_time-start_time,np.array(losses).mean(), lr_scheduler.get_lr()[0]))
        validate(test_dataloader, model, criterion, args, batch_count)


def validate(test_dataloader, model, criterion, args, iter):
    global Class_mIoU, Inst_mIoU, test_dataset
    model.eval()
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()   # initialize augmentation
    seg_classes = test_dataset.seg_classes
    shape_ious = {cat:[] for cat in seg_classes.keys()}
    seg_label_to_cat = {}           # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    losses = 0.0
    lens=len(test_dataloader)
    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            points, target, cls = data
            target = target.cuda()
            points = points.cuda()
            one_hot_target=to_categorical(target,50)
            # augmentation
            #points = PointcloudScaleAndTranslate(points)

            batch_one_hot_cls = np.zeros((len(cls), 16))   # 16 object classes
            for b in range(len(cls)):
                batch_one_hot_cls[b, int(cls[b])] = 1
            batch_one_hot_cls = torch.from_numpy(batch_one_hot_cls).float().cuda()
            # batch_one_hot_cls = Variable(batch_one_hot_cls.float().cuda())

            pred,context = model(points, batch_one_hot_cls)
            pred_t = pred.view(-1, args.num_classes)
            target_t = target.view(-1,1)[:,0]
            loss = criterion(pred_t, target_t,None,context,one_hot_target)
            losses+=loss.item()
            """
            points, target, cls = data
            #points, target = Variable(points, volatile=True), Variable(target, volatile=True)
            points, target = points.cuda(), target.cuda()
            one_hot_target=to_categorical(target,50)
            batch_one_hot_cls = np.zeros((len(cls), 16))   # 16 object classes
            for b in range(len(cls)):
                batch_one_hot_cls[b, int(cls[b])] = 1
            batch_one_hot_cls = torch.from_numpy(batch_one_hot_cls).float().cuda()
            # batch_one_hot_cls = Variable(batch_one_hot_cls.float().cuda())
            pred,context= model(points, batch_one_hot_cls)
            loss = criterion(pred.view(-1, args.num_classes), target.view(-1,1)[:,0],None,context,one_hot_target)
            losses+=loss.item()
            pred = pred.cpu()
            target = target.cpu()
            """
            pred_val = torch.zeros(len(cls), args.num_points).type(torch.LongTensor)
            # pred to the groundtruth classes (selected by seg_classes[cat])
            for b in range(len(cls)):
                cat = seg_label_to_cat[int(target[b, 0].cpu().numpy())]
                logits = pred[b, :, :]   # (num_points, num_classes)
                pred_val[b, :] = logits[:, seg_classes[cat]].max(1)[1] + seg_classes[cat][0]

            for b in range(len(cls)):
                segp = pred_val[b, :].cpu().numpy()
                segl = target[b, :].cpu().numpy()
                cat = seg_label_to_cat[int(segl[0])]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if np.sum((segl == l) | (segp == l)) == 0:
                        # part is not present in this shape
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        instance_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                instance_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_class_ious = np.mean(list(shape_ious.values()))

        for cat in sorted(shape_ious.keys()):
            print('****** %s: %0.6f'%(cat, shape_ious[cat]))
        print('************ Test Loss: %0.6f' % (losses/lens))
        print('************ Class_mIoU: %0.6f' % (mean_class_ious))
        print('************ Instance_mIoU: %0.6f' % (np.mean(instance_ious)))

        if mean_class_ious > Class_mIoU or np.mean(instance_ious) > Inst_mIoU:
            if mean_class_ious > Class_mIoU:
                Class_mIoU = mean_class_ious
            if np.mean(instance_ious) > Inst_mIoU:
                Inst_mIoU = np.mean(instance_ious)
            torch.save(model.state_dict(), '%s/seg_msn_iter_%d_ins_%0.6f_cls_%0.6f.pth' % (args.save_path, iter, np.mean(instance_ious), mean_class_ious))
            #model.train()

if __name__ == "__main__":
    main()
