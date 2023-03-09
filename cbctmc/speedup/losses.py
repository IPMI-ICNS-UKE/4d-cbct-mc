import torch
import torch.nn as nn
import torch.nn.functional as F


def consistency_loss(mean: torch.Tensor, var: torch.Tensor, target: torch.Tensor):
    # shape of mean/var : (batch, 1, x, y)
    gaussian_nll_loss = F.gaussian_nll_loss(
        input=mean, target=target, var=var
    )

    # TV

    tv = torch.mean(torch.abs(torch.diff(mean, dim=0)))

    return 1.0 * gaussian_nll_loss + 10.0 * tv

def calculate_gradient_l2(image: torch.tensor, eps: float = 1e-6) -> torch.tensor:
    x_grad, y_grad = torch.gradient(image, dim=(2, 3))
    l2_grad = torch.sqrt((eps + x_grad ** 2 + y_grad ** 2 ))

    return l2_grad


def gradient_attention_loss(input: torch.tensor, target: torch.tensor):
    input_l2_grad = calculate_gradient_l2(input)
    target_l2_grad = calculate_gradient_l2(target)

    return F.l1_loss(input_l2_grad, target_l2_grad)


def low_preff_loss(mean: torch.Tensor, var: torch.Tensor, target: torch.Tensor):
    gaussian_nll_loss = F.gaussian_nll_loss(
        input=mean, target=target, var=var, reduction="none"
    )
    x = (torch.ones(target.size())*10).to("cuda")
    y = torch.ones(target.size()).to("cuda")
    norm = torch.where(target<4, x,y)


    gaussian_nll_loss = gaussian_nll_loss*norm
    return torch.mean(gaussian_nll_loss)

def low_preff_l1_loss(mean: torch.Tensor, var: torch.Tensor, target: torch.Tensor):
    x = (torch.ones(target.size())*5).to("cuda")
    y = (torch.ones(target.size())).to("cuda")
    norm = torch.where(target<4, x,y)
    l1_loss = F.l1_loss(mean, target, reduce = False)
    l1_loss = l1_loss*norm
    return torch.mean(l1_loss)

def eight_loss(mean: torch.Tensor, var: torch.Tensor, target: torch.Tensor, target2: torch.Tensor):
    gaussian_nll_loss = 0.1 * F.gaussian_nll_loss(
        input=mean, target=target, var=var
    )
    # for i in range(8):
        # gaussian_nll_loss += F.gaussian_nll_loss(input=mean, target=target2[0,i], var=var*2.4/5*100)
    return torch.mean(gaussian_nll_loss)