import torch
import torch.distributed as dist
import torch.nn.functional as F


def masked_mse_loss(pred, target, mask, normalize_targets=False):
    """MSE loss on masked patches

    Args:
        pred (torch.Tensor): B x num_patches x D tensor of predict patches
        target (torch.Tensor): B x num_patches x D tensor of target patch values
        mask (torch.Tensor): B x num_patches binary mask with masked patches marked with 1
    """
    # Normalize target pixel values
    if normalize_targets:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5

    # Calculate MSE loss
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # Per patch loss
    loss = (loss * mask).sum() / mask.sum()  # Mean of masked patches

    return loss


"""
Modified from: 
https://github.com/vturrisi/solo-learn/blob/main/solo/losses/simclr.py
https://github.com/vturrisi/solo-learn/blob/main/solo/utils/misc.py
"""


def info_nce_loss(z: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (2*B) x D tensor containing features from the views.

    Return:
        torch.Tensor: SimCLR loss.
    """

    z = F.normalize(z, dim=-1)
    gathered_z = gather(z)

    sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / temperature)

    indexes = torch.arange(z.size(0) // 2, device=z.device).repeat(2)
    gathered_indexes = gather(indexes)

    indexes = indexes.unsqueeze(0)
    gathered_indexes = gathered_indexes.unsqueeze(0)

    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)

    # negatives
    neg_mask = indexes.t() != gathered_indexes

    pos = torch.sum(sim * pos_mask, 1)
    neg = torch.sum(sim * neg_mask, 1)
    loss = -(torch.mean(torch.log(pos / (pos + neg))))
    return loss


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)
