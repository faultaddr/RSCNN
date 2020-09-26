#coding=utf-8
"""
Author:panyunyi
"""
'''
思路简介：
1.首先构建 context
2.构建 Aggregation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
class Context(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Context,self).__init__()
        self.conv_1=nn.Conv1d(in_channel,in_channel//8,1)
        self.conv_2=nn.Conv1d(in_channel//8,in_channel//16,1)
        # self.conv_3=nn.Conv1d(in_channel//8,in_channel//16,1)
        # self.conv_4=nn.Conv1d(in_channel,in_channel//8,1)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
    def forward(self,x):
        out = nn.functional.relu(self.conv_1(x), inplace=True)
        #context=x.reshape((x.shape[0],x.shape[2],x.shape[2]))
        out_1=self.conv_2(out)
        # out_2=self.conv_3(out)
        context=torch.bmm(out_1.permute(0,2,1),out_1)

        context=self.sigmoid(context)
        p=torch.bmm(context,x.permute(0,2,1))
        p = self.alpha*p
        #p=torch.mul(x,context)
        ones = torch.ones_like(context)
        p_1=torch.bmm(ones-context,x.permute(0,2,1))
        p_1 = self.beta*p
        #p_1=torch.mul(x,ones-context)
        out=torch.cat((x,p.permute(0,2,1),p_1.permute(0,2,1)),dim=1)
        return context,out

class Aggregation(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Aggregation,self).__init__()
        self.conv=nn.Conv1d(in_channel,out_channel,1)
        self.bn=nn.BatchNorm1d(out_channel)
        self.relu=nn.ReLU()
        # self.conv1_1=nn.Conv1d(in_channel,out_channel,(1,k))
        # self.conv1_2=nn.Conv1d(in_channel,out_channel,(k,1))
        # self.conv2_1=nn.Conv1d(in_channel,out_channel,(k,1))
        # self.conv2_2=nn.Conv1d(in_channel,out_channel,(1,k))
        # self.bn1=nn.BatchNorm1d(out_channel)
        # self.relu2=nn.ReLU()

    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        # x_1=self.conv1_1(x,x.shape[1],out_channel)
        # x_1=self.conv1_2(x_1,x_1.shape[1],out_channel)
        # x_2=self.conv2_1(x,x.shape[1],out_channel)
        # x_2=self.conv2_2(x_2,x.shape[1],out_channel)
        # out=torch.cat(x_1,x_2,dim=1)
        return x
"""
class Aggregation(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Aggregation,self).__init__()
        self.conv=nn.Conv1d(in_channel,out_channel,1)
        self.bn=nn.BatchNorm1d(out_channel)
        self.relu=nn.ReLU()
        # self.conv1_1=nn.Conv1d(in_channel,out_channel,(1,k))
        # self.conv1_2=nn.Conv1d(in_channel,out_channel,(k,1))
        # self.conv2_1=nn.Conv1d(in_channel,out_channel,(k,1))
        # self.conv2_2=nn.Conv1d(in_channel,out_channel,(1,k))
        # self.bn1=nn.BatchNorm1d(out_channel)
        # self.relu2=nn.ReLU()

    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        # x_1=self.conv1_1(x,x.shape[1],out_channel)
        # x_1=self.conv1_2(x_1,x_1.shape[1],out_channel)
        # x_2=self.conv2_1(x,x.shape[1],out_channel)
        # x_2=self.conv2_2(x_2,x.shape[1],out_channel)
        # out=torch.cat(x_1,x_2,dim=1)
        return x
"""
def make_one_hot(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y
class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat,context,context_gt):
        context_gt = torch.bmm(context_gt, context_gt.permute(0, 2, 1))
        loss = F.cross_entropy(pred, target)
        #loss=F.nll_loss(pred,target)
        loss_p=F.binary_cross_entropy(context,context_gt)
        loss_g_p = torch.log((context*context_gt).sum(2)+0.01) - torch.log(context.sum(2)+0.01)
        loss_g_r = torch.log((context * context_gt).sum(2)+0.01) - torch.log(context_gt.sum(2)+0.01)
        loss_g_s = torch.log(((1 - context) * (1 - context_gt)).sum(2)+0.01) - torch.log((1 - context_gt).sum(2)+0.01)
        loss_g = - torch.mean(loss_g_p + loss_g_r + loss_g_s)
        #loss_f=F.nll_loss(pred,target)
        #one_hot=make_one_hot(target,pred.size()[1])
        #inv_probs = 1 - pred.exp()
        #focal_weights = (inv_probs * one_hot).sum(dim=1) ** 2
        #loss_fc=loss_f*focal_weights
        #loss_fc=loss_fc.mean()
        total_loss = loss + loss_p + loss_g#+loss_fc
        return total_loss
