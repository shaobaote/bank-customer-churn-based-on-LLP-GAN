import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.constraints import simplex
import torch.autograd as autograd

def cal_gradient_penalty(disc_net, device, real, fake):
    alpha = torch.rand(real.size(0), 1)
    alpha = alpha.expand(real.size())
    alpha = alpha.to(device)
    # 按公式计算x
    interpolates = alpha * real + ((1 - alpha) * fake)
    # 为得到梯度先计算y
    interpolates = autograd.Variable(interpolates, requires_grad=True).to(torch.float32)
    #判别器的输出变了吖
    _, disc_interpolates = disc_net(interpolates)
    # 计算梯度
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,\
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),\
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    # 利用梯度计算出gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def generator_logistic_non_saturating(d_result_fake):
    return F.softplus(-d_result_fake).mean()

def discriminator_logistic_simple_gp(d_result_fake, d_result_real):
    loss = (F.softplus(d_result_fake) + F.softplus(-d_result_real))
    return loss.mean()

def log_sum_exp(x, axis = 1):
    m = torch.max(x, dim = 1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))

def compute_soft_kl(inputs, targets):
    with torch.no_grad():
        loss = cross_entropy_loss(inputs, targets)
        loss = torch.sum(loss, dim=-1).mean()
    return loss


def compute_hard_l1(inputs, targets, num_classes):
    with torch.no_grad():
        predicted = torch.bincount(inputs.argmax(1),
                                   minlength=num_classes).float()
        predicted = predicted / torch.sum(predicted, dim=0)
        targets = torch.mean(targets, dim=0)
        loss = F.l1_loss(predicted, targets, reduction="sum")
    return loss


def cross_entropy_loss(input, target, eps=1e-8):
    #assert simplex.check(input) and simplex.check(target), \
        #"input {} and target {} should be a simplex".format(input, target)
    input = torch.clamp(input, eps, 1 - eps)
    loss = -target * torch.log(input)
    return loss


class ProportionLoss(nn.Module):
    def __init__(self, metric, alpha, eps=1e-8):
        super(ProportionLoss, self).__init__()
        self.metric = metric
        self.eps = eps
        self.alpha = alpha

    def forward(self, input, target):
        # input and target shoud ba a probability tensor
        # and have been averaged over bag size
        assert simplex.check(input) and simplex.check(target), \
            "input {} and target {} should be a simplex".format(input, target)
        assert input.shape == target.shape

        if self.metric == "ce":
            loss = cross_entropy_loss(input, target, eps=self.eps)
        elif self.metric == "l1":
            loss = F.l1_loss(input, target, reduction="none")
        elif self.metric == "mse":
            loss = F.mse_loss(input, target, reduction="none")
        else:
            raise NameError("metric {} is not supported".format(self.metric))

        loss = torch.sum(loss, dim=-1).mean()
        return self.alpha * loss
