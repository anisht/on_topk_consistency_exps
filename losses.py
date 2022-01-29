import numpy as np 
import torch
import torch.nn as nn

def psi2(k):
    def hinge_loss_2(input, target):
        ones = torch.ones(input.shape)
        ones[np.arange(input.shape[0]), target.to(torch.long)] = 0
        onehot_target = 1-ones
        true_label_score = (input*onehot_target).sum(dim=1)
        topk_vals = torch.topk(input+ones, k, dim=1)[0]
        loss = topk_vals.sum(dim=1)/k - true_label_score
        return torch.relu(loss).sum()/input.shape[0]#/tf.cast(tf.shape(y_true)[0], )
    return hinge_loss_2

def psi3(k):
    def hinge_loss_3(input, target):
        ones = torch.ones(input.shape)
        ones[np.arange(input.shape[0]), target.to(torch.long)] = 0
        onehot_target = 1-ones
        true_label_score = (input*onehot_target).sum(dim=1)
        topk_vals = torch.topk(input+ones, k, dim=1)[0]
        loss =(torch.relu(topk_vals- true_label_score[:, None])).sum(dim=1)/k 
        return loss.sum()/input.shape[0]#/tf.cast(tf.shape(y_true)[0], )
    return hinge_loss_3

def psi4(k):
    def hinge_loss_4(input, target): 
        ones = torch.ones(input.shape)
        ones[np.arange(input.shape[0]), target.to(torch.long)] = 0
        onehot_target = 1-ones
        mask = ones.to(torch.bool)
        true_label_score = (input*onehot_target).sum(dim=1)
        topk_vals = torch.topk(input[mask].view(input.size(0), -1), k, dim=1)[0]
        loss = 1+topk_vals.sum(dim=1)/k - true_label_score
        return torch.relu(loss).sum()/input.shape[0]#/tf.cast(tf.shape(y_true)[0], )
    return hinge_loss_4

def psi1(k):
    def l(input, target):
        mask=torch.ones(input.shape).to(torch.bool)
        mask[np.arange(input.shape[0]), target.to(torch.long)]=0
        inputwot = input[mask].view(input.shape[0], input.shape[1]-1)
        if len(inputwot.shape)==1:
            inputwot=inputwot.unsqueeze(0)
        bar = 1+torch.topk(inputwot, k)[0][:, k-1]
        s_y = torch.gather(input, 1, target.view(-1,1).to(torch.long)).flatten()
        return torch.relu(bar-s_y).sum()/input.shape[0]
    return l

def psi6(k):
    def l(input, target):
        # # m1, _ = torch.max(input, 1)
        # # m4 = torch.sum(input, 1) / 4 + 0.25
        # # u_y = torch.gather(input, 1, target.view(-1,1).to(torch.long)).flatten()
        # # losses = torch.maximum(m1, m4) - u_y
        # # return losses.sum()/input.shape[0]


        # # sorted, _ = torch.sort(input, descending=True)  # returns tuple with indices, which we dont need

        # # def m_loss(u, m):
        # #     return (torch.topk(u, m)[0].sum() - min(m, k))/m
        # ms = np.array([1,4,5,6])
        # # ms = np.append(1, range(k + 1, input.shape[1] + 1))
        # m_loss = lambda u, m: (torch.topk(u, m)[0].sum() - min(m, k))/float(m)
        # max_m_loss = lambda u: torch.max(torch.stack([m_loss(u, m) for m in ms]))

        # unbatched = torch.unbind(input)
        # maxes = torch.stack([max_m_loss(u) for u in unbatched])
        # u_y = torch.gather(input, 1, target.view(-1,1).to(torch.long)).flatten()
        # losses = maxes + torch.ones_like(maxes) - u_y
        # return losses.sum()/input.shape[0]

        ms = range(k + 1, input.shape[1] + 1)
        mlosses = [1 - (float(k)/m) + (torch.topk(input, m, dim=1)[0].sum(axis=1))/m  for m in ms] # S_m(u)/m + 1 - k/m for each input row
        max_m_loss, _ = torch.max(torch.stack(mlosses, dim=1), dim=1) # max of m losses

        m1, _ = torch.max(input, dim=1) 
        u_y = torch.gather(input, 1, target.view(-1,1).to(torch.long)).flatten()
        losses = torch.maximum(m1, max_m_loss) - u_y
        return losses.sum()/input.shape[0]





    # print("ASDFL;KAJSDF;LAKJSDF")
    # print(l(torch.tensor([[2, 0, 0, 0]]), torch.tensor([3])))
    return l
# psi6(3)

from torch.autograd import Function
class psi5(Function):
    k=3
    def __init__(self, k=3):
        self.k = k
    @staticmethod
    def forward(ctx, input, target):
        k=psi5.k
        target = target.to(torch.long)
        bar = 1+torch.topk(input, k+1)[0][:,k]
        s_y = torch.gather(input, 1, target.view(-1,1)).flatten()
        ctx.save_for_backward(input, target)
        #ctx.intermediate_results = k
        return torch.relu(bar-s_y).sum()/input.shape[0]

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        #k = ctx.intermediate_results
        k = psi5.k
        target = target.view(-1,1)
        tks, tki = torch.topk(input, k+1)

        tkp1i = tki[:, k].view(-1,1)
        bar = 1+tks[:, k].view(-1,1)
        s_y = torch.gather(input, 1, target)
        mask = (s_y < bar)

        grad_input = torch.zeros(input.shape)
        grad_input[torch.LongTensor(np.arange(grad_input.shape[0]).reshape(-1,1))[mask], tkp1i[mask]] = 1
        grad_input[torch.LongTensor(np.arange(grad_input.shape[0]).reshape(-1,1))[mask], target[mask]]*=0.5
        grad_input[torch.LongTensor(np.arange(grad_input.shape[0]).reshape(-1,1))[mask], target[mask]]+=-1

        return grad_output*grad_input, None, None

# from torch.autograd import Function
# class psi6(Function):
#     k=3
#     def __init__(self, k=3):
#         self.k = k
#     @staticmethod
#     def forward(ctx, input, target):
#         ms = np.append(1, range(k + 1, input.shape[1] + 1))
#         m_loss = lambda u, m: (torch.topk(u, m)[0].sum() - min(m, k))/float(m)
#         max_m_loss = lambda u: torch.max(torch.stack([m_loss(u, m) for m in ms]))

#         unbatched = torch.unbind(input)
#         maxes = torch.stack([max_m_loss(u) for u in unbatched])
#         u_y = torch.gather(input, 1, target.view(-1,1).to(torch.long)).flatten()
#         losses = maxes + torch.ones_like(maxes) - u_y
#         ctx.save_for_backward(input, target)
#         return losses.sum()/input.shape[0]

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, target = ctx.saved_tensors
#         #k = ctx.intermediate_results
#         k = psi6.k
#         target = target.view(-1,1)
#         tks, tki = torch.topk(input, k+1)

#         tkp1i = tki[:, k].view(-1,1)
#         bar = 1+tks[:, k].view(-1,1)
#         s_y = torch.gather(input, 1, target)
#         mask = (s_y < bar)

#         grad_input = torch.zeros(input.shape)
#         grad_input[torch.LongTensor(np.arange(grad_input.shape[0]).reshape(-1,1))[mask], tkp1i[mask]] = 1
#         grad_input[torch.LongTensor(np.arange(grad_input.shape[0]).reshape(-1,1))[mask], target[mask]]*=0.5
#         grad_input[torch.LongTensor(np.arange(grad_input.shape[0]).reshape(-1,1))[mask], target[mask]]+=-1

#         return grad_output*grad_input, None, None

def trent1(k):
    def truncated_entropy_1(input, target):
        n,M = input.shape
        mask = torch.ones(input.shape).to(torch.bool)
        mask[np.arange(n), target.to(torch.long)] = 0
        s_noty = input[mask].view(n, M-1)
        s_y = input[np.arange(n), target.to(torch.long)].view(n, 1)
        
        softmax = nn.Softmax(dim=1)
        
        # careful that rows are in increasing order
        botmmk = -torch.topk(-s_noty, M-k)[0]
        g = softmax(torch.cat((s_y, botmmk), dim=1))
        
        return -torch.log(g[:, 0]).mean()
    return truncated_entropy_1

def trent2(k):
    def truncated_entropy_2(input, target):
        n,M = input.shape
        mask = torch.ones(input.shape).to(torch.bool)
        mask[np.arange(n), target.to(torch.long)] = 0
        s_noty = input[mask].view(n, M-1)
        s_y = input[np.arange(n), target.to(torch.long)].view(n, 1)
        
        softmax = nn.Softmax(dim=1)
        
        # careful that rows are in increasing order
        botmmk = -torch.topk(-s_noty, M-k)[0]
        g = softmax(torch.cat((s_y, botmmk), dim=1))
        
        loss=0
        sum_exp = torch.exp(botmmk).sum(dim=1).view(-1, 1)
        exp_s = torch.exp(input)
        loss = exp_s/(exp_s + sum_exp)
        loss = loss.sum(dim=1).mean()
        
        return loss-torch.log(g[:, 0]).mean()-1
    return truncated_entropy_2