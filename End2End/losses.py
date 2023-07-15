from functools import partial
import torch
from End2End.constants import VELOCITY_SCALE
import numpy as np
import itertools

def l1_wav(output: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    r"""L1 loss in the time-domain.

    Args:
        output: torch.Tensor
        target: torch.Tensor

    Returns:
        loss: torch.float
    """
    return torch.mean(torch.abs(output - target))


def get_loss_function(loss_types):
    return partial(versatile_loss_function, loss_types=loss_types)


def versatile_loss_function(model, output_dict, target_dict, loss_types: list = None):
    """a loss function that can be adapted based on the requeste loss types.

    `model` argument is not needed. it is kept to be compatible with the existing code but should be removed later.

    Args:
        model (nn.Module)
        output_dict (dict): dict of model outputs
        target_dict (dict): dict of targets
        loss_types (list): list of str that specifies each loss type

    """
    if len(loss_types) == 0 or loss_types is None:
        raise ValueError('At least one loss type is required.')

    loss = 0.0
    for loss_type in loss_types:

        if loss_type == 'onset':
            loss += bce(output_dict['reg_onset_output'], target_dict['reg_onset_roll'], target_dict['mask_roll'])

        elif loss_type == 'offset':
            loss += bce(output_dict['reg_offset_output'], target_dict['reg_offset_roll'], target_dict['mask_roll'])

        elif loss_type == 'frame':
            loss += bce(output_dict['frame_output'], target_dict['frame_roll'], target_dict['mask_roll'])

        elif loss_type == 'velocity':
            loss += bce(
                output_dict['velocity_output'],
                target_dict['velocity_roll'] / VELOCITY_SCALE,
                target_dict['onset_roll'],
            )

        else:
            raise NotImplementedError(f'loss type ({loss_type}) is not defined yet.')

    return loss


def bce(output, target, mask):
    r"""Binary crossentropy (BCE) with mask. The positions where mask=0 will be
    deactivated when calculation BCE.
    """
    eps = 1e-7
    output = torch.clamp(output, eps, 1.0 - eps)
    matrix = -target * torch.log(output) - (1.0 - target) * torch.log(1.0 - output)
    return torch.sum(matrix * mask) / torch.sum(mask)

'''
def pit_loss(output, target, mixture_roll):

    def _loss_func(o, t, m):
        eps = 1e-7
        o = torch.clamp(o, eps, 1. - eps)
        matrix = - t * torch.log(o) - (1. - t) * torch.log(1. - o)
        return torch.sum(matrix * m) / torch.clamp(torch.sum(m), eps, np.inf)

    def _loss_func2(o, t, m):
        eps = 1e-7
        o = torch.clamp(o, eps, 1. - eps)
        matrix = - t * torch.log(o) - (1. - t) * torch.log(1. - o)
        from IPython import embed; embed(using=False); os._exit(0)
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2,1, sharex=True)
        axs[0].matshow(o.data.cpu().numpy().T, origin='lower', aspect='auto', cmap='jet')
        axs[1].matshow(t.data.cpu().numpy().T, origin='lower', aspect='auto', cmap='jet')
        plt.savefig('_zz.pdf')
        return torch.sum(matrix * m) / torch.clamp(torch.sum(m), eps, np.inf)

    (batch_size, clusters_num, frames_num, pitches_num) = target.shape
    # F.binary_cross_entropy(output, target)

    total_loss = 0.

    for n in range(batch_size):

        indexes_list = list(range(clusters_num))
        best_locts = []

        for i in range(clusters_num):

            _losses = []
            for j in indexes_list:
                _loss = _loss_func(output[n, i], target[n, j], mixture_roll[n]).item()
                _losses.append(_loss)
            loct = np.argmin(_losses)
            best_locts.append(indexes_list[loct])

            indexes_list.pop(loct)

        for i in range(clusters_num):
            _loss = _loss_func(output[n, i], target[n, best_locts[i]], mixture_roll[n])
            total_loss += _loss
            # print(n, _loss)

        # if n == 11:
        #     _loss_func2(output[n, i], target[n, best_locts[i]], mixture_roll[n])
            # from IPython import embed; embed(using=False); os._exit(0)


    total_loss = total_loss / (batch_size * clusters_num)

    return total_loss
'''


def pit_loss(output, target, mixture_roll):

    def _loss_func(o, t, m):
        eps = 1e-7
        o = torch.clamp(o, eps, 1. - eps)
        matrix = - t * torch.log(o) - (1. - t) * torch.log(1. - o)
        return torch.sum(matrix * m) / torch.clamp(torch.sum(m), eps, np.inf)

    (batch_size, clusters_num, frames_num, pitches_num) = target.shape
    # F.binary_cross_entropy(output, target)

    total_loss = 0.

    for n in range(batch_size):

        loss_mat = np.zeros((clusters_num, clusters_num))

        for i in range(clusters_num):
            for j in range(clusters_num):
                loss_mat[i, j] = _loss_func(output[n, i], target[n, j], mixture_roll[n]).item()

        permutations = list(itertools.permutations(range(clusters_num)))
        loss_list = []
        for permutation in permutations:
            _loss = 0.
            for i in range(clusters_num):
                _loss += loss_mat[i, permutation[i]]
            loss_list.append(_loss)

        loct = np.argmin(loss_list)
        best_matches = permutations[loct]

        _total_loss = 0.
        for i in range(clusters_num):
            _total_loss += _loss_func(output[n, i], target[n, best_matches[i]], mixture_roll[n])

        total_loss += _total_loss

    total_loss = total_loss / (batch_size * clusters_num)

    return total_loss


'''
def pit_loss(output, target):

    def _loss_func(o, t):
        eps = 1e-7
        o = torch.clamp(o, eps, 1. - eps)
        matrix = - t * torch.log(o) - (1. - t) * torch.log(1. - o)
        return torch.mean(matrix)

    (batch_size, clusters_num, frames_num, pitches_num) = target.shape
    # F.binary_cross_entropy(output, target)

    total_loss = 0.

    for n in range(batch_size):

        indexes_list = list(range(clusters_num))
        best_locts = []

        for i in range(clusters_num):

            _losses = []
            for j in indexes_list:
                _loss = _loss_func(output[n, i], target[n, j]).item()
                _losses.append(_loss)
            loct = np.argmin(_losses)
            best_locts.append(indexes_list[loct])

            indexes_list.pop(loct)

        for i in range(clusters_num):
            _loss = _loss_func(output[n, i], target[n, best_locts[i]])
            total_loss += _loss

    total_loss = total_loss / (batch_size * clusters_num)

    return total_loss
'''

def nll_loss(output, target):
    loss = - torch.mean(target * output)
    return loss


def l1_loss(output, target):
    return torch.mean(torch.abs(output - target))